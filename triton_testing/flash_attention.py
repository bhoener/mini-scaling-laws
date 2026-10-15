import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@triton.jit
def flash_attn_forward(Q_ptr, K_ptr, V_ptr, N, d, Q_stride_N, Q_stride_d, K_stride_N, K_stride_d, V_stride_N, V_stride_d, BLOCK_SIZE_COL: tl.constexpr, BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_D: tl.constexpr):
    """
    Do flash attention here

    We divide Q into T_r = ceil(N/B_r) blocks of size B_r x d
    K, V into T_c = ceil(N/B_c) blocks
    """
    pid = tl.program_id(axis=0)
    T_r = tl.cdiv(N, BLOCK_SIZE_ROW)
    T_c = tl.cdiv(N, BLOCK_SIZE_COL)

    for i in range(T_r):
        # (B_r) -> (B_r, 1)
        # (B_d) -> (1, B_d)
        # (B_r, 1) + (1, B_d) -> (B_r, B_d)
        offsets_Q = (tl.arange(0, BLOCK_SIZE_ROW)[:, None] + pid * BLOCK_SIZE_ROW) * Q_stride_N + tl.arange(0, BLOCK_SIZE_D)[None, :] * Q_stride_d
        Q_mask = (tl.arange(0, BLOCK_SIZE_ROW) + pid * BLOCK_SIZE_ROW < N)[:, None] & (tl.arange(0, BLOCK_SIZE_D) < d)[None, :]
        Q_i = tl.load(Q_ptr + offsets_Q, mask=Q_mask, other=0.0)

        O = tl.zeros((BLOCK_SIZE_ROW, d), dtype=tl.float32)

        denom = tl.zeros((BLOCK_SIZE_ROW,), dtype=tl.float32)

        for j in range(T_c):
            if i <= j:
                offsets_K = (tl.arange(0, BLOCK_SIZE_COL)[:, None] + pid * BLOCK_SIZE_COL) * K_stride_N + tl.arange(0, BLOCK_SIZE_D)[None, :] * K_stride_d 
                K_mask = (tl.arange(0, BLOCK_SIZE_COL) + pid * BLOCK_SIZE_ROW < N)[:, None] & (tl.arange(0, BLOCK_SIZE_D) < d)[None, :]
                K_j = tl.load(K_ptr + offsets_K, mask=K_mask, other=0.0)
                offsets_V = (tl.arange(0, BLOCK_SIZE_COL)[:, None] + pid * BLOCK_SIZE_COL) * K_stride_N + tl.arange(0, BLOCK_SIZE_D)[None, :] * K_stride_d 





class FlashAttention(torch.autograd.Function):
    def forward(ctx, Q, K, V, is_causal=True):
        pass
    def backward(ctx, dO):
        pass