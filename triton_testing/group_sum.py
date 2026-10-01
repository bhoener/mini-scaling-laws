import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfgs = []
for block_size in [1024, 2048, 4096]:
    for warps in [2, 4, 8]:
        cfgs.append(triton.Config({"BLOCK_SIZE": block_size}, num_warps=warps))

@triton.autotune(
    configs=cfgs, key={"M", "N"}
)
@triton.jit
def _group_sum_kernel_1(in_ptr, intermediate_ptr, locks_ptr, M, N, GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Performs the first part of a group sum reduction
    (M, N) -> (GROUP_SIZE, N)

    keep in mind that the name GROUP_SIZE is misleading.
    GROUP_SIZE is (M // ROWS_PER_GROUP)

    our locks are of size (GROUP_SIZE * 2)

    ok so:

    for each row, we must find its corresponding row in the output
    we also have locks. why are there GROUP_SIZE * 2 locks?
    not sure.

    """

    row = tl.program_id(axis=0)
    intermediate_row = row % GROUP_SIZE

    row_start = row * N
    offsets = tl.arange(0, BLOCK_SIZE_N)

    mask = offsets < N

    row_contribution = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)


    # so, looks like:
    # we have a locks tensor
    # it is (2 * GROUP_SIZE)
    # containing (is_locked, has_been_accessed)

    # if it is our first time accessing, we don't read the intermediate
    # and just write
    # but, if it is not our first time accessing,
    # we add the current intermediate value to our contribution
    # and then *overwrite*. we do not add the addition.
    # this is the same as adding our contribution to the intermediate value
    # since addition is commutative (bruh)

    pid_lock_ptr = locks_ptr + intermediate_row
    # since locks is of form [lock, lock, ..., accessed, accessed]
    pid_lock_accessed_ptr = locks_ptr + intermediate_row + GROUP_SIZE

    intermediate_block_start = intermediate_ptr + intermediate_row * N
    intermediate_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask_intermediate = intermediate_offsets < N

    while tl.atomic_cas(pid_lock_ptr, 0, 1) == 1:
        pass

    count = tl.load(pid_lock_accessed_ptr)
    if count == 0:
        tl.atomic_xchg(pid_lock_accessed_ptr, 1)
    else:
        row_contribution += tl.load(intermediate_block_start + intermediate_offsets, mask=mask_intermediate, other=0.0)

    tl.store(intermediate_block_start + intermediate_offsets, row_contribution, mask=mask_intermediate)

@triton.autotune(
    configs=cfgs, key={"M", "N"}
)
@triton.jit
def _group_sum_kernel_2(intermediate_ptr, out_ptr, GROUP_SIZE: tl.constexpr, N, BLOCK_SIZE: tl.constexpr):
    """
    Now we just sum along cols

    [ |  |  |  | ]
    [ |  |  |  | ]
    [ |  |  |  | ]
    [ V  V  V  V ] => [ x  x  x  x ]

    this might be slop, idk i don't have a gpu to check it since hpc maintenance
    """
    col = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE) * N + col

    mask = tl.arange(0, BLOCK_SIZE) < GROUP_SIZE


    col_data = tl.load(intermediate_ptr + offsets, mask=mask, other=0.0)

    sum_res = tl.sum(col_data, axis=0)

    tl.store(out_ptr + col, sum_res)





def group_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous
    x = x.to(torch.float32)
    x = x.to(DEVICE)
    M, N = x.size()

    GROUP_SIZE = 256

    inter = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=DEVICE)

    locks = torch.zeros(GROUP_SIZE * 2, dtype=torch.int32, device=DEVICE)

    _group_sum_kernel_1[(M,)](x, inter, locks, GROUP_SIZE)

    out = torch.empty(N, device=DEVICE, dtype=torch.float32)

    _group_sum_kernel_2[(N,)](inter, out, GROUP_SIZE, N)

    return out


if __name__ == "__main__":
    x = torch.randn(24, 32, device=DEVICE, dtype=torch.float32)

    x_summed = group_sum(x)

    x_summed_ref = x.sum(dim=0)

    print(x_summed)
    print(x_summed_ref)

    torch.testing.assert_close(x_summed, x_summed_ref, atol=1e-2, rtol=0)
    
    print("PASSED")
