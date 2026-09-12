import torch
import triton
import triton.language as tl

# https://www.youtube.com/watch?v=sDSvJ6juB2I

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# apparently there exists a better method
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGISTERS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]  # smallest possible group of cores
print("num sm:", NUM_SM)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x_block = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

    expon = tl.exp(x_block)


def softmax(x: torch.Tensor) -> torch.Tensor:

    # why are we doing all this extra stuff for softmax?
    # is there something different that the kernel requires?
    # or is it just to demonstrate manual tuning?

    assert x.ndim == 2
    n_rows, n_cols = x.size()

    # we will assume everything fits in SRAM

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # why are we specifying this?
    # doesnt each pid take BLOCK_SIZE elements?
    # ok figured it out
    # BLOCK_SIZE stays the same
    # but normally threads < BLOCK_SIZE
    # so threads do operations multiple times
    # if BLOCK_SIZE=1024, num_warps=4, warp_size=32,
    # we do 1024 / (4 * 32) = 8 operations per thread
    # triton does abstraction for us! yay!

    # we set block size bc we have limited sram

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 2048:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2  # do multiple things @ once?

    y = torch.empty_like(x, device=x.device)

    kernel = softmax_kernel.warmup(
        x,
        y,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )

    kernel._init_handles()

    n_regs_per_thread = kernel.n_regs
    sram_per_program = kernel.metadata.shared

    # how many programs can we fit on an SM?
    reg_occupancy = NUM_REGISTERS // (n_regs_per_thread * WARP_SIZE * num_warps)

    # programs/sm = (regs/sm) / (regs/thread * threads/warp * warps/program)

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_per_program

    programs_per_sm = min(reg_occupancy, sram_occupancy)

    num_programs = programs_per_sm * NUM_SM

    num_programs = min(
        num_programs, n_rows
    )  # now we are hoping everything fits into BLOCK_SIZE
    # and using our program axis to determine the row computed

    grid = (
        num_programs,
        1,
        1,
    )  # apparently need to specify the extra axes? will learn later

    # x.stride(dim) = num of steps forward to take in memory to get to next entry along dim
    # if x is MxN
    # x.stride() = (M, 1)
    # x.stride(0) = M
    # x.stride(1) = 1

    kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
