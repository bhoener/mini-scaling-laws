import torch
import triton
import triton.language as tl
# import os
# os.environ["TRITON_INTERPRET"] = "1"

# NOTE: works fine on ampere/higher but very, very slow on turing

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
print("using device", DEVICE)

autotune_config = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
]


@triton.autotune(configs=autotune_config, key=["M", "K", "N"])
@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: int,
    K: int,
    N: int,
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
    # BLOCK_SIZE_N = 128 cols/pid
    # BLOCK_SIZE_K = 64 {rows, cols}/pid
    # GROUP_SIZE=2 pid/group

    # M = N = K = 512 {rows | cols}

    pid = tl.program_id(axis=0)

    # 512 rows / 128 rows/pid = 4 pids
    num_pid_along_m = tl.cdiv(M, BLOCK_SIZE_M)
    # 512 rows / 64 rows/pid = 8 pids
    num_pid_along_n = tl.cdiv(N, BLOCK_SIZE_N)
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
    # group_id = 21 // (8 * 2) = 1
    # first_pid_in_group_along_m = group_id * GROUP_SIZE = 1 * 2 = 2
    # pid_m = 2 + ((21 % 8) % 2) = 2 + (5 % 2) = 2 + 1 = 3
    pid_m = first_pid_in_group_along_m + ((pid % num_pid_in_group) % group_size_adj)
    # pid_n = (21 % 8) // 2 = 5 // 2 = 2
    pid_n = (pid % num_pid_in_group) // group_size_adj

    # what are offsets_M?
    """
    [        ...                    ]
              :
    [pid_m * BLOCK_SIZE_M ...       ]
              :
    [pid_m * (BLOCK_SIZE_M + 1) ... ]
              :
    [        ...                    ]

    but how do we convert this to a 1d array we can load?
    """
    offsets_M = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    """
    say we have
    [ 0  1  2  3 ]
    [ 4  5  6  7 ]
    [ 8  9 10 11 ]
    [12 13 14 15 ]
    offsets_m = [ 2  3 ] -> [[ 2 ] * 4  =  [[ 8  ]
                             [ 3 ]]         [ 12 ]]
    offsets_k = [ 0  1  2  3 ] -> [[ 0  1  2  3 ]]
    -> [[ 8  9 10 11 ]]
       [[12 13 14 15 ]]
    """
    x_offsets = offsets_M[:, None] * x_row_stride + offsets_K[None, :] * x_col_stride
    y_offsets = offsets_K[:, None] * y_row_stride + offsets_N[None, :] * y_col_stride

    # remember that M, N are fixed for each pid
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K
        # why don't we mask along M/N?
        # i think group_size_adj manages when there aren't enough pids to form a full group
        # but what about when there aren't enough elements to form a block?

        # triton infers the shape of x and y using the shape of the pointer arrays
        x = tl.load(x_ptr + x_offsets, mask=mask[None, :], other=0.0)
        y = tl.load(y_ptr + y_offsets, mask=mask[:, None], other=0.0)

        accumulator = tl.dot(x, y, acc=accumulator)

        x_offsets += BLOCK_SIZE_K * x_col_stride
        y_offsets += BLOCK_SIZE_K * y_row_stride

    out_offsets = offsets_M[:, None] * out_row_stride + offsets_N[None, :] * out_col_stride

    out_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)

    tl.store(out_ptr + out_offsets, accumulator.to(tl.float16), mask=out_mask)


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and y.ndim == 2 and x.size(1) == y.size(0)

    (M, K), (_, N) = x.size(), y.size()

    x = x.to(torch.float16)
    y = y.to(torch.float16)
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

    print(c_tri)
    print(c_ref)

    assert torch.allclose(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")
    

test_matmul_kernel((512, 512))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128*i for i in range(2, 16)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPs",
        plot_name="torch-triton-matmul",
        args={},
    )
)
def benchmark(M: int, N: int, K: int, provider: str) -> float:
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    y = torch.randn(K, N, dtype=torch.float16, device=DEVICE)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x, y), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(x, y), quantiles=quantiles)

    tflops = lambda ms: M * N * 3 * K / 1e12 / (ms / 1e3)
    return tflops(ms), tflops(min_ms), tflops(max_ms)

benchmark.run(show_plots=False, print_data=True, save_path=".")