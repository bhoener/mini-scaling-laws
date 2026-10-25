import torch
import os

os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl\

import numpy as np
np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cfgs = []
# for bsc in [64, 128, 256]:
#     for bsr in [64, 128, 256]:
#         cfgs.append(
#             triton.Config(
#                 {"BLOCK_SIZE_COL": bsc, "BLOCK_SIZE_ROW": bsr}
#             )
#         )


# @triton.autotune(configs=cfgs, key={"N", "d"})
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

    """

    # start with the pid and some constants
    pid = tl.program_id(axis=0)
    T_r = tl.cdiv(N, BLOCK_SIZE_ROW)
    T_c = tl.cdiv(N, BLOCK_SIZE_COL)

    print("d:", d)

    
    # (B_r) -> (B_r, 1)
    # (B_d) -> (1, B_d)
    # (B_r, 1) + (1, B_d) -> (B_r, B_d)
    offsets_Q = (
        (tl.arange(0, BLOCK_SIZE_ROW)[:, None]
         + pid * BLOCK_SIZE_ROW) * Q_stride_N
        + tl.arange(0, d)[None, :] * Q_stride_d
    )

    # mask and load Q
    Q_mask = (tl.arange(0, BLOCK_SIZE_ROW) + pid * BLOCK_SIZE_ROW < N)[:, None] & (
        tl.arange(0, d) < d
    )[None, :]
    Q_i = tl.load(Q_ptr + offsets_Q, mask=Q_mask, other=0.0)

    # initialize empty output
    O = tl.zeros((BLOCK_SIZE_ROW, d), dtype=tl.float32)

    # denominators l
    denom = tl.zeros((BLOCK_SIZE_ROW,), dtype=tl.float32)

    # row-wise maxes
    m = tl.full((BLOCK_SIZE_ROW,), float("-inf"), dtype=tl.float32)

    for j in range(T_c):
        # we can skip if j > i since causal
        if not (j * BLOCK_SIZE_COL >= (pid + 1) * BLOCK_SIZE_ROW):
            # create offsets for K
            offsets_K = (
                (tl.arange(0, BLOCK_SIZE_COL)[:, None] + j * BLOCK_SIZE_COL)
                * K_stride_N
                + tl.arange(0, d)[None, :] * K_stride_d
            )

            #print("offsets_K:", offsets_K)

            # mask is same for k and v
            KV_mask = (tl.arange(0, BLOCK_SIZE_COL) + j * BLOCK_SIZE_COL < N)[
                :, None
            ] & (tl.arange(0, d) < d)[None, :]

            # load K
            K_j = tl.load(K_ptr + offsets_K, mask=KV_mask, other=0.0)


            # create offsets for V
            offsets_V = (
                (tl.arange(0, BLOCK_SIZE_COL)[:, None] + j * BLOCK_SIZE_COL)
                * V_stride_N + tl.arange(0, d)[None, :] * V_stride_d
            )
            #print("offsets_V:", offsets_V)

            # load V
            V_j = tl.load(V_ptr + offsets_V, mask=KV_mask, other=0.0)

            # Calculate Q_i K_j^T
            # (B_r, d) x (d, B_c) -> (B_r, B_c)
            
            S = tl.dot(Q_i, tl.trans(K_j))
            
            if pid * BLOCK_SIZE_ROW < (j + 1) * BLOCK_SIZE_COL:
                causal_mask_bool = (
                    tl.arange(0, BLOCK_SIZE_ROW)[:, None] + (pid * BLOCK_SIZE_ROW)
                    < tl.arange(0, BLOCK_SIZE_COL)[None, :] + (j * BLOCK_SIZE_COL)
                )
                causal_mask = tl.where(causal_mask_bool, float("-inf"), 0)

                # apply casual mask
                S = S + causal_mask

            # store running maxes
            m_last = m

            m = tl.maximum(m, tl.max(S, axis=1))

            # numerator of softmax
            P_temp = tl.exp(S - m[:, None])


            # update denominator (need to read more into online softmax)
            # denom is (B_r,)
            # m is (B_r,)
            # P_temp is (B_r, B_c)
            denom = tl.exp(m_last - m) * denom + tl.sum(P_temp, axis=1)

            # calculate block output
            # P_temp is (B_r, B_c), V is (B_c, d)
            # PV is (B_r, d)

            # O is (B_r, d)
            # m is (B_r,)
            O = O * tl.exp(m_last - m)[:, None] + tl.dot(P_temp, V_j)

            # print(f"i={pid}, j={j}", "K:", K_j, "V:", V_j, "S:", S, "M_last:", m_last, "M:", m, "P_temp:", P_temp, "denom:", denom, "O:", O, sep="\n")
    # if pid * BLOCK_SIZE_ROW <= j * BLOCK_SIZE_COL:
    #     print("K:", K_j, "V:", V_j, "S:", S, "M_last:", m_last, "M:", m, "P_temp:", P_temp, "denom:", denom, "O:", O, sep="\n")
    O = O / denom[:, None]
    L = m + tl.log(denom)

    tl.store(O_ptr + offsets_Q, O, mask=Q_mask)
    tl.store(L_ptr + tl.arange(0, BLOCK_SIZE_ROW), L)


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

    grid = lambda meta: (
        tl.cdiv(N, meta["BLOCK_SIZE_ROW"]),
    )

    print("K stride N:", K.stride(0), "K stride d:", K.stride(1))

    BLOCK_SIZE_ROW = 4
    BLOCK_SIZE_COL = 8
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
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    return O


def main():
    T, C = 64, 8
    Q = torch.randn(T, C, device=DEVICE, dtype=torch.float32)
    K = torch.randn(T, C, device=DEVICE, dtype=torch.float32)
    V = torch.randn(T, C, device=DEVICE, dtype=torch.float32)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("reference:", ref)
    tri = _flash_attn_forward(Q, K, V)
    print("tri:", tri)

    print(ref == tri)
    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=0)

    print("Passed test")


if __name__ == "__main__":
    main()


class FlashAttention(torch.autograd.Function):
    def forward(ctx, Q, K, V, is_causal=True):
        pass

    def backward(ctx, dO):
        pass
