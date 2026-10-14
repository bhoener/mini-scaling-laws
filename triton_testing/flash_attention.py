import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@triton.jit
def flash_attn_forward(Q_ptr, K_ptr, V_ptr, N, d, Q_stride_N, Q_stride_d, K_stride_N, K_stride_d, V_stride_N, V_stride_d, BLOCK_SIZE: tl.constexpr):
    """
    Do flash attention here
    """

class FlashAttention(torch.autograd.Function):
    def forward(ctx, Q, K, V, is_causal=True):
        pass
    def backward(ctx, dO):
        pass