import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfgs = []
for block_size in [512, 1024, 2048]:
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

    ok so:

    for each row, we must find its corresponding row in the output
    we also have locks. why are there GROUP_SIZE * 2 locks?
    not sure.

    """

    row = tl.program_id(axis=0)
    inter_row = row % GROUP_SIZE

    # now what
    # need to look into atomic operations



def group_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous
    x = x.to(torch.float32)
    x = x.to(DEVICE)
    M, N = x.size()

    GROUP_SIZE = 256

    inter = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=DEVICE)

    locks = torch.zeros(GROUP_SIZE * 2, type=torch.int32, device=DEVICE)

    _group_sum_kernel_1[(M,)](x, inter, locks, GROUP_SIZE)



