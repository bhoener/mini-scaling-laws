import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def dot_kernel(x_ptr,
               y_ptr,
               out_ptr,
               num_elements: int,
               BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < num_elements

    x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_block = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    output = tl.sum(x_block * y_block, axis=0)

    tl.atomic_add(out_ptr, output)

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros(1, device=DEVICE, dtype=torch.float32)

    assert x.device == DEVICE and y.device == DEVICE and z.device == DEVICE

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    dot_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=1024)

    return z

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(28)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        plot_name="triton-torch-dot",
        ylabel="GFLOPS",
        args={},
    )
)
def benchmark(size: int, provider: str) -> tuple[float]:
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles=[0.5, 0.2, 0.8]

    if provider=="torch":
        median, min_ms, max_ms = triton.testing.do_bench(lambda: torch.dot(x, y), quantiles=quantiles)
    if provider=="triton":
        median, min_ms, max_ms = triton.testing.do_bench(lambda: dot(x, y), quantiles=quantiles)

    gflops = lambda ms: (x.numel() * 3 / (ms / 1e3)) / 1e9

    return gflops(median), gflops(min_ms), gflops(max_ms)

benchmark.run(show_plots=True, print_data=True)

x = torch.rand(2048, device=DEVICE)
y = torch.rand(2048, device=DEVICE)

z_triton = dot(x, y)
z_torch = torch.dot(x, y)

print(z_triton)
print(z_torch)