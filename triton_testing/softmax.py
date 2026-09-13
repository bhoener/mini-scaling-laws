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
def softmax_kernel(x_ptr, out_ptr, x_row_stride: int, out_row_stride: int, n_rows: int, n_cols: int, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_step = tl.num_programs(0)
    # if 4 programs, row_step=4
    # pid0 row 0
    # pid1 row 1
    # pid2 row 2
    # pid3 row 3
    # once done w/ first assigned row
    # pid0 -> row 4 (+= row_step)

    # this is if programs < rows
    # we start with row_idx=pid, then go to row_idx = pid+row_step if needed
    # num_stages is to parallelize
    for row_idx in tl.range(pid, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = x_ptr + row_idx * x_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)

        mask = col_offsets < n_cols

        x_row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float("-inf"))

        row_minus_max = x_row - tl.max(x_row, axis=0)

        numerator = tl.exp(row_minus_max)

        output = numerator / tl.sum(numerator, axis=0)

        tl.store((out_ptr + row_idx * x_row_stride) + col_offsets, output, mask)


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
        x.stride(0),
        y.stride(0),
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

    # contiguous is when stride is as expected (prod of previous dimensions), but can be
    # changed by view(), transpose(), permute() ...

    # don't have to pass tl.constexpr (eg. BLOCK_SIZE, num_stages, num_warps) if done in warmup
    kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        x.size(0),
        x.size(1),
    )

    return y

def test_softmax(size: tuple[int, ...], atol: float = 1e-3, rtol: float = 1e-3, device=DEVICE) -> None:
    x = torch.randn(size, device=device)

    torch_softmax = torch.nn.functional.softmax(x, dim=-1)
    kernel_softmax = softmax(x)

    print(torch_softmax)
    print(kernel_softmax)

    assert torch.allclose(torch_softmax, kernel_softmax, atol=atol, rtol=rtol)
    print("PASSED")

test_softmax((4, 8))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128*i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles = [("blue", "-"), ("green", "-")],
        plot_name="triton-torch-softmax",
        args={"M": 4096},
        ylabel="GB/s",
    )
)
def benchmark(M: int, N: int, provider: str) -> tuple[float]:
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    if provider == "torch":
        median = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
    if provider == "triton":
        median = triton.testing.do_bench(lambda: softmax(x))

    gbps = lambda ms: (x.element_size() * x.numel() / 1e9) / (ms / 1e3)
    return gbps(median)

benchmark.run(show_plots=True, print_data=True, save_path=".")