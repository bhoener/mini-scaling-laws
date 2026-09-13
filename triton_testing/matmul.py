import torch
import triton
import triton.language as tl


"""
for m in range(0, M, BLOCK_SIZE_M):
    for n in range(0, N, BLOCK_SIZE_N):
        acc = tl.zeros(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m:m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]
            b = B[k:k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N]
            acc += tl.dot(a, b)
        C[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE:N] = acc

ok i finally realized
block matmul is just matmul on matrices

      [E | F]
      -------
      [G | H]
[A | B]      =[AE + BG  |  AF + BH]
-------       ---------------------
[C | D]       [CE + DG  |  CF + DH]
"""

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@triton.jit
def matmul_kernel(x_ptr,
                  y_ptr,
                  out_ptr,
                  m: int,
                  n: int,
                  p: int,
                  x_row_stride: int,
                  x_col_stride: int,
                  y_row_stride: int,
                  y_col_stride: int,
                  out_row_stride: int,
                  out_col_stride: int,
                  BLOCK_SIZE: tl.constexpr,
                  num_stages: tl.constexpr):
    pass

def test_matmul_kernel(size: tuple[int, ...], atol=1e-3, rtol=1e-1) -> None:
    

