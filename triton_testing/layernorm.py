import torch
import triton
import triton.language as tl
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@triton.jit
def _layernorm_forward(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, stride_M, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        # load a single row of x
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask = cols < N, other=0.0).to(tl.float32)

        


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        M, N = x.reshape(-1, x.shape(-1)).shape
        y = torch.empty_like(x, device=DEVICE, dtype=torch.float32)
        mean = torch.empty((M, ), device=DEVICE, dtype=torch.float32)
        rstd = torch.empty((M, ), device=DEVICE, dtype=torch.float32)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("this layer norm doesn't support feature dime >= 64kb")

        # normally want 1 warp per 256 data but limit to <= 8 and >= 1
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layernorm_forward[(M,)](
            x, y, weight, bias, mean, rstd, 
            x.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

    @staticmethod
    def backward( ):

def test_layernorm_kernel(M: int, N: int, dtype: torch.dtype, eps: float = 1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=DEVICE)
    x.requires_grad_(True)
    weight = torch.rand((N,), dtype=dtype, device=DEVICE, requires_grad=True)
    bias = torch.randn((N,), dtype=dtype, device=DEVICE, requires_grad=True)
    y_tri = layernorm(x, (N,), weight, bias, eps)
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

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    print("passed backward")