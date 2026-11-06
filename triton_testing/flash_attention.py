import torch
import os
import triton
import triton.language as tl
import numpy as np

np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfgs = []
for bsc in [64, 128]:
    for bsr in [64, 32]:
        cfgs.append(
            triton.Config(
                {"BLOCK_SIZE_COL": bsc, "BLOCK_SIZE_ROW": bsr}
            )
        )


@triton.autotune(configs=cfgs, key={"N", "d"})
@triton.jit
def _flash_attn_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    O_ptr,
    N,
    d: tl.constexpr,
    Q_stride_N,
    Q_stride_d,
    K_stride_N,
    K_stride_d,
    V_stride_N,
    V_stride_d,
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    We divide Q into T_r = ceil(N/B_r) blocks of size B_r x d
    K, V into T_c = ceil(N/B_c) blocks

    Why B_r, B_c?

    We have Q:
                d
        [ x x x x x x ]
        [ x x x x x x ] B_r
        ---------------
      N [ x x x x x x ]
        [ x x x x x x ]
        ---------------
        [ x x x x x x ]
        [ x x x x x x ]

    K, V:
        [ x x x x x x ]
        [ x x x x x x ] B_c
        [ x x x x x x ]
      N ---------------
        [ x x x x x x ]
        [ x x x x x x ]
        [ x x x x x x ]

    Q = [ A ]
        [ B ]
        [ C ]

    K = [ D ]
        [ E ]

    QK^T =        [ D^T   E^T ]
            [ A ] [ AD^T AE^T ]
            [ B ] [ BD^T BE^T ]
            [ C ] [ CD^T CE^T ]


    this is T_r x T_c

    But if we individually transpose the blocks, do we get a different result?

    It seems like it could be the same, since each block of q contains B_r query tokens
    and each block of k contains B_c key tokens

    so qk^T should fit somewhere within everything

    what if we define out to be T_r x T_c empty

    [ x x ]
    [ x x ]
    [ x x ]

    then out[i, j] = q_i k_j^T

    yes, this is the same. we don't have to deal with transposing the blocks or anything since
    the blocks are all along one dimension

    this is very hard to debug. there is nothing obviously wrong in any of the stages.

    the offsets look okay and all the intermediate values are plausible.

    only the first row is correct, which means nearly anything could be broken

    interestingly, it is the first row of the first block that is correct
    and not the whole block.

    this means P_temp @ V_j is correct for the first row of P and first column of V

    which means at least one of the following:

    - one of the rows of m is wrong
    - one of the rows of S is wrong

    (assuming V_j is correct)


    ok, what are we doing?

                    K^T
             [ x  x  x  x  x ]
        [ x ][ T  T  T  T  T ]
        [ x ][ T  T  T  T  T ]
    Q   [ x ][ T  T  T  T  T ]
        [ x ][ T  T  T  T  T ]
        [ x ][ T  T  T  T  T ]

    interesting. the first two rows of our output are almost identical.
    why is this? all indices look good.

    nevermind. this has stopped happening (?)

    notably, the first row of the output is the same as the first row of V.
    q_i^T k_j has some random value, but we softmax with all n=1..j so it just becomes 1,
    which weights v_j.
                        [ x ]
                        [ x ]
                        [ x ]
                    k_j [ x ]
                        [ x ]
           q_i^T        [ x ]
    [ x  x  x  x  x  x ][ y ] * [ x  x  x  x  x  x ]

    and with SM(QK^T)V it is

               [ x  x ]     V
            K^T[ x  x ][ x  x  x ]
          Q    [ x  x ][ x  x  x ]
    [ x  x  x ][ y  y ][ z  z  z ]
    [ x  x  x ][ y  y ][ z  z  z ]

    note that each y is q_i^T k_j and this is multiplied and summed by v to get z

    
    this is probably why the code is correct for the first row.

    something about the online softmax or the calculation of S is likely wrong.

    i also checked whether turning off is_causal had any impact. with it off,
    i got zero correct rows. this makes sense since my one row is only correct because
    it does not get any contributions from previous tokens.

    this means my causal mask is (probably) not the issue

    when i set q and k to all zeros, things seem to work correctly.

    this seems to imply there is an issue in calculating QK^T

    when i set Q[1] = ones, K[1] = ones, v[1] = zeros,
    i get a row of zeros in my triton output
    while the reference gives it a row of 0.06

    could this actually be a causal mask issue?
                     [ 0  1  0  0 ]
                     [ 0  1  0  0 ]                     [v11 v12 v13 v14]
                     [ 0  1  0  0 ]                     [v21 v22 v23 v24]
                     [ 0  1  0  0 ]                     [v31 v32 v33 v34]
                     [ 0  1  0  0 ]                     [ ...     ...   ]
    [ 0  0  0  0  0 ][ 0  -  -  - ]   [1    0   0   0 ] [v11 v12 v13 v14]
    [ 1  1  1  1  1 ][ 0  d  -  - ] ->[~0  ~1   0   0 ]~[v21 v22 v23 v24]
    [ 0  0  0  0  0 ][ 0  0  0  - ]   [... ... ...  0 ]     :     :
    [ 0  0  0  0  0 ][ 0  0  0  0 ]   [... ... ... ...] [ ...     ...   ]

    the one row of v is all zeros, so the little bit must
    be coming from the row before it

    note that the inner product of q_1 and k^T_0 is 0
    so the softmax is producing the little bit

    interestingly, one time it gave the output -1.64, -1.64 for the second row.
    i have not since been able to reproduce this.

    is it the denominator that is incorrect? all the rows after the row of zeros are good.
    
    even if i set B_r to 2, the values are correct.


    """

    # start with the pid and some constants
    pid = tl.program_id(axis=0)
    T_r = tl.cdiv(N, BLOCK_SIZE_ROW)
    T_c = tl.cdiv(N, BLOCK_SIZE_COL)


    # (B_r) -> (B_r, 1)
    # (B_d) -> (1, B_d)
    # (B_r, 1) + (1, B_d) -> (B_r, B_d)
    offsets_Q = (
        tl.arange(0, BLOCK_SIZE_ROW)[:, None] + pid * BLOCK_SIZE_ROW
    ) * Q_stride_N + tl.arange(0, BLOCK_SIZE_D)[None, :] * Q_stride_d

    # mask and load Q
    Q_mask = (tl.arange(0, BLOCK_SIZE_ROW) + pid * BLOCK_SIZE_ROW < N)[:, None] & (
        tl.arange(0, BLOCK_SIZE_D) < d
    )[None, :]
    Q_i = tl.load(Q_ptr + offsets_Q, mask=Q_mask, other=0.0)

    # initialize empty output
    O = tl.zeros((BLOCK_SIZE_ROW, BLOCK_SIZE_D), dtype=tl.float32)

    # denominators l
    denom = tl.zeros((BLOCK_SIZE_ROW,), dtype=tl.float32)

    # row-wise maxes
    m = tl.full((BLOCK_SIZE_ROW,), float("-inf"), dtype=tl.float32)

    for j in range(T_c):
        # we can skip if j > i since causal
        if not (j * BLOCK_SIZE_COL >= (pid + 1) * BLOCK_SIZE_ROW):
            # create offsets for K
            offsets_K = (
                tl.arange(0, BLOCK_SIZE_COL)[:, None] + j * BLOCK_SIZE_COL
            ) * K_stride_N + tl.arange(0, BLOCK_SIZE_D)[None, :] * K_stride_d

            #print("offsets_K:", offsets_K)

            # mask is same for k and v
            KV_mask = (tl.arange(0, BLOCK_SIZE_COL) + j * BLOCK_SIZE_COL < N)[
                :, None
            ] & (tl.arange(0, BLOCK_SIZE_D) < d)[None, :]

            # load K
            K_j = tl.load(K_ptr + offsets_K, mask=KV_mask, other=0.0)

            # create offsets for V
            offsets_V = (
                tl.arange(0, BLOCK_SIZE_COL)[:, None] + j * BLOCK_SIZE_COL
            ) * V_stride_N + tl.arange(0, BLOCK_SIZE_D)[None, :] * V_stride_d

            # load V
            V_j = tl.load(V_ptr + offsets_V, mask=KV_mask, other=0.0)

            # Calculate Q_i K_j^T
            # (B_r, d) x (d, B_c) -> (B_r, B_c)

            S = tl.dot(Q_i, tl.trans(K_j)) * (d ** -0.5)
            

            if pid * BLOCK_SIZE_ROW < (j + 1) * BLOCK_SIZE_COL:
                causal_mask_bool = tl.arange(0, BLOCK_SIZE_ROW)[:, None] + (
                    pid * BLOCK_SIZE_ROW
                ) < tl.arange(0, BLOCK_SIZE_COL)[None, :] + (j * BLOCK_SIZE_COL)
                causal_mask = tl.where(causal_mask_bool, float("-inf"), 0)

                # apply casual mask
                S = S + causal_mask


            # store running maxes
            m_last = m

            m = tl.maximum(m, tl.max(S, axis=1))

            # numerator of softmax
            P_temp = tl.exp(S - m[:, None])

            scale = tl.exp(m_last - m)

            # update denominator (need to read more into online softmax)
            # denom is (B_r,)
            # m is (B_r,)
            # P_temp is (B_r, B_c)
            denom = scale * denom + tl.sum(P_temp, axis=1)

            # calculate block output
            # P_temp is (B_r, B_c), V is (B_c, d)
            # PV is (B_r, d)

            # O is (B_r, d)
            # m is (B_r,)
            # what is this actually calculating?
            # similar to https://arxiv.org/pdf/1805.02867 algorithm 3
            # denom = denom * exp(m_last - m) + exp(x_i - m)
            #       = denom * exp(m_last) / exp(m) + exp(x_i) / exp(m)
            # for example, j=0
            # denom = 0
            # j = 1
            # m = -inf
            # denom = 0 * exp(-inf - 0) + exp(x_1) / exp(0) = exp(x_1)
            # j = 2
            # note that in this example, x_j is is scalar
            # m = max(m, m_last) = max(x_2, x_1)
            # denom = exp(x_j) * exp(x_1 - max(x_2, x_1)) + exp(x_2 - max(x_2, x_1))

            # interestingly, we are calculating O at the same time as the denominator
            # while in the above paper, they calculate the two separately

            # for intuition, it seems like the scale factor rescales O by
            # removing the previous scale via multiplication and adding the
            # new scale via division.

            O = O * scale[:, None] + tl.dot(P_temp, V_j)

    O = O / denom[:, None]
    L = m + tl.log(denom)

    tl.store(O_ptr + offsets_Q, O, mask=Q_mask)
    tl.store(L_ptr + pid * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW), L)


def _flash_attn_forward(Q, K, V):
    N, d = Q.size()
    assert Q.size() == K.size() == V.size()
    Q, K, V = (
        Q.to(DEVICE).to(torch.float32),
        K.to(DEVICE).to(torch.float32),
        V.to(DEVICE).to(torch.float32),
    )
    O = torch.empty_like(Q)
    L = torch.empty(N, device=DEVICE, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_ROW"]),)


    _flash_attn_forward_kernel[grid](
        Q,
        K,
        V,
        L,
        O,
        N=N,
        d=d,
        Q_stride_N=Q.stride(0),
        Q_stride_d=Q.stride(1),
        K_stride_N=K.stride(0),
        K_stride_d=K.stride(1),
        V_stride_N=V.stride(0),
        V_stride_d=V.stride(1),
        BLOCK_SIZE_D=triton.next_power_of_2(d)
    )

    return O


def online_softmax(x: torch.Tensor):
    assert x.ndim == 2
    out = torch.empty_like(x)

    
    for i in range(x.size(0)):
        m = float("-inf")
        d = 0
        for j in range(x.size(1)):
            m_last = m
            m = max(m, x[i, j])
            d = d * torch.exp(m_last - m) + torch.exp(x[i, j] - m)

        for j in range(x.size(1)):
            out[i, j] = torch.exp(x[i, j] - m) / d

    return out


def main():
    T, C = 64, 18
    Q = torch.randn(T, C, device=DEVICE, dtype=torch.float32)
    K = torch.randn(T, C, device=DEVICE, dtype=torch.float32)
    V = torch.randn(T, C, device=DEVICE, dtype=torch.float32)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("reference:", ref)
    tri = _flash_attn_forward(Q, K, V)
    print("tri:", tri)
    print("ref:", ref)

    x = torch.randn(24, 32)
    
    softmax_test = online_softmax(x)
    softmax_ref = torch.softmax(x, dim=-1)

    torch.testing.assert_close(softmax_test, softmax_ref, atol=1e-2, rtol=0)
    print("online softmax passed")
    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=0)

    print("Passed test")

    benchmark.run(show_plots=False, print_data=True, save_path=".")

 
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128*i for i in range(2, 64)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        plot_name="flash_attn_perf",
        args={"d": 16},
        xlabel=["sequence length"],
        ylabel="tflops",
        styles=[("green", "-"), ("blue", "-")]
    )
)
def benchmark(N: int, d: int, provider: str) -> float | tuple[float, ...]:
    Q = torch.randn(N, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
    K = torch.randn(N, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
    V = torch.randn(N, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        with torch.no_grad():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _flash_attn_forward(Q, K, V), quantiles=quantiles)

    tflops = lambda ms: 4 * N ** 2 * d / 1e12 / (ms * 1e-3)
    return tflops(ms), tflops(min_ms), tflops(max_ms)

if __name__ == "__main__":
    main()


class FlashAttention(torch.autograd.Function):
    def forward(ctx, Q, K, V, is_causal=True):
        pass

    def backward(ctx, dO):
        pass
