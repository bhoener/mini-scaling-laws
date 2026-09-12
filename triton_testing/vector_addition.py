import time
import torch
import triton
import triton.language as tl

# following https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

dv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(dv)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# 1) @triton.jit above function
# 2) takes pointers to x, y, and output, no return
# 3) takes n_elements
# 4) BLOCK_SIZE


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):

    # seems like there are multiple programs, each deals with its own data
    pid = tl.program_id(axis=0)  # axis=0 bc 1d launch grid

    # each program accesses a block of data, with a unique offset

    # pid 0 -> start 0, pid 1 -> start BLOCK_SIZE, pid 2 -> start 2*BLOCK_SIZE...
    block_start = pid * BLOCK_SIZE  # int
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # list of pointers

    mask = offsets < n_elements  # boolean mask so no out-of-bounds

    # load from DRAM
    x_block = tl.load(x_ptr + offsets, mask=mask) # list of ptrs
    y_block = tl.load(y_ptr + offsets, mask=mask)

    # x_block and y_block are literally just tensors

    output = x_block + y_block

    # where is output stored, though?

    # sounds like once we do tl.load, it goes from VRAM -> SRAM
    # SRAM very fast but very small

    # store back in DRAM
    # takes arguments: (list_of_ptrs, data, mask)
    tl.store(
        output_ptr + offsets, output, mask=mask
    )  # note that output_ptr + offsets is a list of ptrs

    # no return

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ok this is pretty much what i would expect, do an empty allocation
    output = torch.empty_like(x, device=x.device)
    n_elements = output.numel()

    # make sure everything is on same device, also makes sense
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    # now we construct a launch grid
    # must either be tuple[int] or Callable(metaparameters) -> tuple[int]
    # i think in our case since we are in 1D, we need a 1-element tuple
    # containing the number of blocks needed

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # kernel[grid](...)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # return z but kernel still running async
    return output

# now use triton's builtin benchmarking utilities

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"], # argument name for x axis, eg. fn(size=x)
        x_vals=[2**i for i in range(28)],
        x_log=True, # logarithmic x
        line_arg="provider", # argument name corresponding to different lines in the plot
        line_vals=["triton", "torch"], # values to set line_arg to
        line_names=["Triton", "Torch"], # corresponding names for plot
        styles=[("blue", "-"), ("green", "-")], # corresponding styles
        ylabel="GB/s",
        plot_name="torch-triton-vector-add", # plot and file name
        args={}, # values for function args not in x_names and y_name
    )
)
def benchmark(size: int, provider: str) -> tuple[float]:
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8] # percentiles for execution time, returns same size

    if provider=="triton":
        median, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    if provider=="torch":
        median, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e9 / (ms * 1e-3)
    return gbps(median), gbps(min_ms), gbps(max_ms)

benchmark.run(print_data=True, show_plots=True)