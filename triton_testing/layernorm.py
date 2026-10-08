import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@triton.jit
def _layernorm_forward(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    stride_M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        # load a single row of x
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        sum_accumulator += x
    mean = tl.sum(sum_accumulator, axis=0) / N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        diff = tl.where(cols < N, x - mean, 0.0)
        acc += diff * diff

    var = acc.sum(axis=0) / N
    rstd = 1 / (var + eps).sqrt()

    # store mean, rstd for backward
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        weight_block = tl.load(w_ptr + cols, mask=cols < N, other=0.0)
        bias_block = tl.load(b_ptr + cols, mask=cols < N, other=0.0)
        x_block = tl.load(x_ptr + cols, mask=cols < N, other=0.0)

        x_shifted = (x_block - mean) * rstd * weight_block + bias_block

        tl.store(y_ptr + cols, x_shifted, mask=cols < N)


@triton.jit
def _layernorm_backward_dLdx(x_ptr, dLdx_ptr, dLdy_ptr, w_ptr, dLdw_inter_ptr, dLdb_inter_ptr, mean_ptr, rstd_ptr, locks_ptr, x_stride_M, N, GROUP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Pretty much just following the math from https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html,
    though i've probably done something wrong

    interesting. the first row is correct, but all the other rows are wrong.
    this suggests something is wrong with my pointer math
    """

    # each pid represents a row
    row = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    offsets += x_stride_M * row

    # load components we need
    x_row = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    rstd = tl.load(rstd_ptr + row)
    mean = tl.load(mean_ptr + row)

    # recalculate xhat, probably faster than transferring
    x_hat = (x_row - mean) * rstd
    x_hat = tl.where(mask, x_hat, 0.0)

    dLdy_row = tl.load(dLdy_ptr + offsets, mask=mask, other=0.0)

    w_row = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    w_dLdy = w_row * dLdy_row

    c1 = tl.sum(x_hat * w_dLdy, axis=0) / N
    c2 = tl.sum(w_dLdy, axis=0) / N

    # calculate the gradient row
    dLdx_row = rstd * (w_dLdy - c1 * x_hat - c2)
    tl.store(dLdx_ptr + offsets, dLdx_row, mask=mask)

    # create our contributions for the row
    dLdw_contribution = dLdy_row * x_hat
    dLdb_contribution = dLdy_row

    # pointer math for group sum
    intermediate_row = row % GROUP_SIZE

    intermediate_row_start = intermediate_row * N

    intermediate_row_offsets = tl.arange(0, BLOCK_SIZE)

    mask_intermediate = intermediate_row_offsets < BLOCK_SIZE

    intermediate_row_offsets += intermediate_row_start

    lock_ptr = locks_ptr + intermediate_row
    lock_used_ptr = lock_ptr + GROUP_SIZE

    # while in use, wait
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    # if it's not been accessed yet, do nothing
    # otherwise add our contributions
    used = tl.load(lock_used_ptr)
    if used == 0:
        tl.atomic_xchg(lock_used_ptr, 1)
    else:
        dLdw_contribution += tl.load(dLdw_inter_ptr + intermediate_row_offsets, mask=mask_intermediate, other=0.0)
        dLdb_contribution += tl.load(dLdb_inter_ptr + intermediate_row_offsets, mask=mask_intermediate, other=0.0)

    # store back and unlock
    tl.store(dLdw_inter_ptr + intermediate_row_offsets, dLdw_contribution, mask=mask_intermediate)
    tl.store(dLdb_inter_ptr + intermediate_row_offsets, dLdb_contribution, mask=mask_intermediate)

    tl.atomic_xchg(lock_ptr, 0)

@triton.jit
def _layernorm_backward_dLdw_dLdb(dLdw_inter_ptr, dLdb_inter_ptr, dLdw_ptr, dLdb_ptr, GROUP_SIZE: tl.constexpr, N, BLOCK_SIZE: tl.constexpr):
    # each PID is a column
    # might be faster to have a pid be multiple columns
    col = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < GROUP_SIZE

    col_offsets = offsets * N + col

    # load, sum, and store
    dLdw_inter_col = tl.load(dLdw_inter_ptr + col_offsets, mask=mask, other=0.0)
    dLdb_inter_col = tl.load(dLdb_inter_ptr + col_offsets, mask=mask, other=0.0)

    dLdw_contribution = tl.sum(dLdw_inter_col, axis=0)
    dLdb_contribution = tl.sum(dLdb_inter_col, axis=0)

    tl.store(dLdw_ptr + col, dLdw_contribution)
    tl.store(dLdb_ptr + col, dLdb_contribution)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        M, N = x.reshape(-1, x.size(-1)).shape
        y = torch.empty_like(x, device=DEVICE, dtype=torch.float32)
        mean = torch.empty((M,), device=DEVICE, dtype=torch.float32)
        rstd = torch.empty((M,), device=DEVICE, dtype=torch.float32)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("this layer norm doesn't support feature dime >= 64kb")

        # normally want 1 warp per 256 data but limit to <= 8 and >= 1
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layernorm_forward[(M,)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            x.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.size(-1)).shape

        # initialize empty gradients
        dLdx = torch.empty_like(x)  # (M, N)
        dLdw = torch.empty_like(w)  # (N)
        dLdb = torch.empty_like(b)  # (N)

        GROUP_SIZE = 64
        if N <= 8192:
            GROUP_SIZE = 96
        if N <= 4096:
            GROUP_SIZE = 128
        if N <= 1024:
            GROUP_SIZE = 256

        # make intermediate tensors (why GROUP_SIZE, N?)
        # is GROUP_SIZE the size of the reduced tensor?
        dLdw_inter = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=x.device)
        dLdb_inter = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=x.device)

        locks = torch.zeros((2 * GROUP_SIZE,), dtype=torch.int32, device=x.device)

        _layernorm_backward_dLdx[(M,)](
            x,
            dLdx,
            dLdy,
            w,
            dLdw_inter,
            dLdb_inter,
            mean,
            rstd,
            locks,
            x.stride(0),
            N,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        _layernorm_backward_dLdw_dLdb[(N,)](
            dLdw_inter,
            dLdb_inter,
            dLdw,
            dLdb,
            min(GROUP_SIZE, M),
            N,
            BLOCK_SIZE=ctx.BLOCK_SIZE
        )

        return dLdx, None, dLdw, dLdb, None

layer_norm = LayerNorm.apply

def test_layernorm_kernel(
    M: int, N: int, dtype: torch.dtype, eps: float = 1e-5, device=DEVICE
):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=DEVICE)
    x.requires_grad_(True)
    weight = torch.rand((N,), dtype=dtype, device=DEVICE, requires_grad=True)
    bias = torch.randn((N,), dtype=dtype, device=DEVICE, requires_grad=True)
    y_tri = layer_norm(x, (N,), weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps).to(dtype)

    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)
    print("passed forward")

    # ensure gradients are correct
    dLdy = 0.1 * torch.randn_like(x)
    y_tri.backward(dLdy, retain_graph=True)
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]


    print(dLdx_tri)
    print(dLdx_ref)
    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    print("passed backward")

# def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
#     # create data
#     x_shape = (M, N)
#     w_shape = (x_shape[-1], )
#     weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
#     bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
#     x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
#     dy = .1 * torch.randn_like(x)
#     x.requires_grad_(True)
#     # forward pass
#     y_tri = layer_norm(x, w_shape, weight, bias, eps)
#     y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
#     # backward pass (triton)
#     y_tri.backward(dy, retain_graph=True)
#     dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
#     x.grad, weight.grad, bias.grad = None, None, None
#     # backward pass (torch)
#     y_ref.backward(dy, retain_graph=True)
#     dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
#     # compare
#     print(dx_tri, dx_ref)
#     assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
#     assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
#     assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
#     assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)

def main():
    test_layernorm_kernel(256, 256, torch.float32)

if __name__ == "__main__":
    main()