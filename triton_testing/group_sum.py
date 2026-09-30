import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfgs = []
for warps in [2, 4, 8]:
    cfgs.append(triton.Config(num_warps=warps))

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

    while tl.atomic_cas(pid_lock_ptr, 0, 1) == 1:
        pass

    count = tl.load(pid_lock_accessed_ptr)
    if count == 0:
        tl.atomic_xchg(pid_lock_accessed_ptr, 1)
    else:
        row_contribution += tl.load() # load intermediate row here
    



def group_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous
    x = x.to(torch.float32)
    x = x.to(DEVICE)
    M, N = x.size()

    GROUP_SIZE = 256

    inter = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=DEVICE)

    locks = torch.zeros(GROUP_SIZE * 2, type=torch.int32, device=DEVICE)

    _group_sum_kernel_1[(M,)](x, inter, locks, GROUP_SIZE)



