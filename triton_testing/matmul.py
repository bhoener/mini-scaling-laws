import torch
import triton
import triton.language as tl
# import os
# os.environ["TRITON_INTERPRET"] = "1"


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

autotune_config = [
    triton.config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=4,
    ),
    triton.config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=4,
    ),
    triton.config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE": 8},
        num_stages=3,
        num_warps=4,
    ),
]


@triton.autotune(configs=autotune_config, key=["m", "k", "n"])
@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    m: int,
    k: int,
    n: int,
    x_row_stride: int,
    x_col_stride: int,
    y_row_stride: int,
    y_col_stride: int,
    out_row_stride: int,
    out_col_stride: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    M = N = K = 8
    BLOCK_SIZE_{M, K, N} = 2
    [0,  1,  2,  3]
    [4,  5,  6,  7]
    [8,  8, 10, 11]
    [12,13, 14, 15]

    PID0
    [x, x, x, x]    [x, _, _, _]
    [_, _, _, _]    [x, _, _, _]
    [_, _, _, _]    [x, _, _, _]
    [_, _, _, _]    [x, _, _, _]

    PID1
    [x, x, x, x]    [_, x, _, _]
    [_, _, _, _]    [_, x, _, _]
    [_, _, _, _]    [_, x, _, _]
    [_, _, _, _]    [_, x, _, _]

    ...

    PID 0-3 all read the first row, but we have to read all of B

    PID4
    [_, _, _, _]    [x, _, _, _]
    [x, x, x, x]    [x, _, _, _]
    [_, _, _, _]    [x, _, _, _]
    [_, _, _, _]    [x, _, _, _]

    PID5
    [_, _, _, _]    [_, x, _, _]
    [x, x, x, x]    [_, x, _, _]
    [_, _, _, _]    [_, x, _, _]
    [_, _, _, _]    [_, x, _, _]

    We need
    [0,  1, |  4,  5]
    [2,  3, |  6,  7]
    -----------------
    [8,  9, | 12, 13]
    [10,11, | 14, 15]

    """

    # let BLOCK_SIZE_M = 128 rows/pid
    # BLOCK_SIZE_N = 64 cols/pid
    # GROUP_SIZE=2 pid/group

    # M = N = K = 512 {rows | cols}

    pid = tl.program_id(axis=0)

    # 512 rows / 128 rows/pid = 4 pids
    num_pid_along_m = tl.cdiv(m, BLOCK_SIZE_M)
    # 512 rows / 64 rows/pid = 8 pids
    num_pid_along_n = tl.cdiv(n, BLOCK_SIZE_N)
    # 2 pid/group * 8 pids = 16 pid^2/group
    num_pid_in_group = GROUP_SIZE * num_pid_along_n
    # pid / (pid / group) = group
    group_id = pid // num_pid_in_group

    # group * pid/group = pid
    first_pid_in_group_along_m = group_id * GROUP_SIZE

    group_size_adj = min(num_pid_along_m - first_pid_in_group_along_m, GROUP_SIZE)

    """
      ------------8------------
    | [ 0  1  2  3  4  5  6  7 ]
    | [ 8  9 10 11 12 13 14 15 ]
    4 [16 17 18 19 20 21 22 23 ]
    | [24 25 26 27 28 29 30 31 ]
    
                  |
                  V
    
      --------------8-------------
    | [ 0  1 | 4  5 | 8  9 |12 13 ]
    | [ 2  3 | 6  7 |10 11 |14 15 ]
    4  -----  -----  -----  -----
    | [16 17 |20 21 |24 25 |28 29 ]
    | [18 19 |22 23 |26 27 |30 31 ]

    WRONG!!!!

    i have been using (M, N) x (N, P)
    and the example is (M, K) x (K, N)

    all this is for the output matrix

    
                                  -------4------
                                | [ 0  1 |  2  3 ]
                                | [ 4  5 |  6  7 ]
                                | [ 8  9 | 10 11 ]
                                8 [12 13 | 14 15 ]
                                | [16 17 | 18 19 ]
                                | [20 21 | 22 23 ]
                                | [24 25 | 26 27 ]
                                | [28 29 | 30 31 ]
                                    ^ groups ^
      ------------8------------
    | [ 0  1  2  3  4  5  6  7 ]  \
    | [ 8  9 10 11 12 13 14 15 ]  /
    4 --------------------------  > groups
    | [16 17 18 19 20 21 22 23 ]  \
    | [24 25 26 27 28 29 30 31 ]  /

    in our output we have

    [ 0  1 | 4  5 ]
    [ 2  3 | 6  7 ]
    ------- -------
    [ 8  9 | 12 13]
    [10 11 | 14 15]

    these are groups, but they are not related to block matmul. just for loading efficiency

    in order to compute (0, 0), we need r0 of A and c0 of B
    (0, 0) -> A[0, :], B[:, 0] (where A[i, j] is a block of size (BLOCK_SIZE_M x BLOCK_SIZE_K))
    (0, 1) -> A[0, :], B[:, 1]
    (1, 0) -> A[1, :], B[:, 0]
    (1, 1) -> A[1, :], B[:, 1]

    
    """

    # say pid=21
    # num_pid_along_m=4
    # num_pid_along_n=8
    # group_id = 21 // (8 * 2) = 1 (?)
    # first_pid_in_group_along_m = group_id * GROUP_SIZE = 1 * 8 = 8
    # pid_m = 8 + ((9 % 8) % 2) = 8 + (1 % 2) = 8 + 1 = 9
    pid_m = first_pid_in_group_along_m + ((pid % num_pid_in_group) % group_size_adj)
    pid_n = pid % num_pid_in_group


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and y.ndim == 2 and x.size(1) == y.size(0)

    (M, K), (_, N) = x.size(), y.size()

    z = torch.empty(M, N, device=DEVICE, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )  # (if 3x4 BLOCK_SIZE_M=BLOCK_SIZE_N=4, grid=(16,))

    # why no warmup?
    matmul_kernel[grid](
        x,
        y,
        z,
        M,
        K,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        z.stride(0),
        z.stride(1),
    )

    return z


def test_matmul_kernel(
    size: tuple[int, ...], atol=1e-3, rtol=1e-1, device=DEVICE
) -> None:
    assert type(size) == tuple and len(size) == 2
    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)

    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)

    assert torch.allclose(c_tri, c_ref, atol=atol, rtol=rtol)
