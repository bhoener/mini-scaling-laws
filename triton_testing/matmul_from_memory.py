import torch
import os
os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl



DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE": 2}, num_stages=3)
]

@triton.autotune(configs=configs, key={"M", "K", "N"})
@triton.jit
def matmul_kernel(a_ptr,
                  b_ptr,
                  c_ptr,
                  M: int,
                  K: int,
                  N: int,
                  a_stride_M: int,
                  a_stride_K: int,
                  b_stride_K: int,
                  b_stride_N: int,
                  c_stride_M: int,
                  c_stride_N: int,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE: tl.constexpr) -> None:
    """                   B       N
                          [ 0  1 |  2  3 |  4 ]
                          [ 5  6 |  7  8 |  9 ]
                        K [10 11 | 12 13 | 14 ]
                          [15 16 | 17 18 | 19 ]
                          [20 21 | 22 23 | 24 ]
                          [25 26 | 27 28 | 29 ]
      A        K          C                    
      [ 0  1  2  3  4  5 ][ 0  1 | 4   5 | 8  9 ]
      [ 6  7  8  9 10 11 ][ 2  3 | 6   7 | 10 11]
    M ------------------- ----------------------
      [12 13 14 15 16 17 ][12 13 | 16 17 | 20 21]
                          

    what are our PIDs actaully doing?

    each pid:

    acc = zeros(BLOCK_SIZE_M, BLOCK_SIZE_N)
    for k in range(blocks_K):
        x_block = load_block(x, pid_M, k) # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        y_block = load_block(y, k, pid_N) # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        acc += tl.dot(x_block, y_block)

    recall:
                [y, y, _, _]
                [y, y, _, _]
                [y, y, _, _]
                [y, y, _, _]
    [x, x, x, x][z, z, _, _]
    [x, x, x, x][z, z, _, _]
    [_, _, _, _][_, _, _, _]
    [_, _, _, _][_, _, _, _]

    i remember before there was an example where pid 21 mapped to group 1 in a 4*8 example.
    there was also something like group_size_m = GROUP_SIZE * num_pids_along_m

    hmm

    units:
    pids/group * pids = pids^2/group

    which i guess makes sense
    but a group is not along all rows of the first matrix
    that would defeat the purpose

    ok maybe i should stop trying to remember the method used and derive it myself

    given GROUP_SIZE, BLOCK_SIZE_{M, K, N}, M, K, N, pid, find pid_M, pid_N

    we can use division to find blocks_{M, K, N}

    we need to find what group the pid is in. converting group to coordinates isn't very hard

    how to find group id? probably just pid // group_size_in_pids
    but must be careful
    since group_size 2 either corresponds to group_size_M = 2, group_size_N = 2 or group_size_total = 2
    im guessing its the first since we need a square anyway

    so any reason not to just do pid // GROUP_SIZE**2?
    i'm hesitant because i remember it not being that

    but will go with it anyways

    i rememeber there was also an adjustment for when there weren't enough blocks to form a full group
    not sure why masking couldn't handle that

    it was like group_size_adj = min(something - something else, something).

    whatever. 

    """
    pid = tl.program_id(axis=0)

    # get number of blocks along each axis
    blocks_M = tl.cdiv(M, BLOCK_SIZE_M)
    blocks_K = tl.cdiv(K, BLOCK_SIZE_K)
    blocks_N = tl.cdiv(N, BLOCK_SIZE_N)

    groups_M = tl.cdiv(blocks_M, GROUP_SIZE)
    groups_N = tl.cdiv(blocks_N, GROUP_SIZE)

    # print(f"Groups M: {groups_M}, Groups N: {groups_N}") 
    # we must add a full block for matmul but don't necessarily need a full group (like 5x5 -> don't need 6x6 blocks)

    # we need to find the group start for the given PID
    # to do this, we need to find the size of a group and the group_id
    group_id = pid // GROUP_SIZE ** 2
    start_of_group = group_id * GROUP_SIZE ** 2

    
    pid_m = ((pid - start_of_group) // GROUP_SIZE + GROUP_SIZE * (group_id // groups_N))
    # subtract start of group to get id relative to group
    # mod by group size to get the n component and then add back group M
    pid_n = ((pid - start_of_group) % GROUP_SIZE + GROUP_SIZE * (group_id % groups_N))

    print(f"pid {pid} -> ({pid_m}, {pid_n})")
    # seems to work fine on 512x512

    # great. now we just need to do the indices and masking and then the kernel
    # TODO: look into why we reassign the accumulator after tl.dot(a, b, acc=accumulator)

    offsets_M = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = offsets_M[:, None] * a_stride_M + offsets_K[None, :] * a_stride_K
    b_offsets = offsets_K[:, None] * b_stride_K + offsets_N[None, :] * b_stride_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(blocks_K):
        max_k_block = K - k * BLOCK_SIZE_K

        k_mask = offsets_K < max_k_block

        # a_chunk is (M, K)
        # k_mask is (K), so we need to index [None, :] -> (1, K)
        a_chunk = tl.load(a_ptr + a_offsets, mask=k_mask[None, :], other=0.0)
        b_chunk = tl.load(b_ptr + b_offsets, mask=k_mask[:, None], other=0.0)

        

        accumulator = tl.dot(a_chunk, b_chunk, acc=accumulator)

        # now increment the offsets
        # if we start at block (0, 0), we will go to (0, 1)
        # so a offsets need to take a step in the column blocks

        a_offsets += BLOCK_SIZE_K * a_stride_K
        b_offsets += BLOCK_SIZE_K * b_stride_K

    
    c_offsets = offsets_M[:, None] * c_stride_M + offsets_N[None, :] * c_stride_N
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    print(c_mask)
    tl.store(c_ptr + c_offsets, accumulator.to(tl.float16), mask=c_mask)






def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    (M, K), (_, N) = a.size(), b.size()
    assert a.size(1) == b.size(0)
    assert a.device == b.device
    a = a.to(torch.float16)
    b = b.to(torch.float16)

    c = torch.empty(M, N, device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    matmul_kernel[grid](
        a, b, c,
        M, K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c

def test_matmul(M: int, K: int, N: int, atol=1e-2, rtol=1e-1, device=DEVICE, dtype=torch.float16) -> None:
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype) / (K ** 0.5)

    c_torch = torch.matmul(a, b)
    c_triton = matmul(a, b)

    print(c_torch)
    print(c_triton)
    torch.testing.assert_close(c_torch, c_triton, atol=atol, rtol=rtol)
    print("PASSED")

if __name__ == "__main__":
    print("using device", DEVICE)
    test_matmul(513, 513, 513)