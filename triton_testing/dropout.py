import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs=[]
for block_size in [32, 64, 128, 256, 512, 1024, 2048]:
    for warps in [2, 4]:
        for num_stages in range(1, 4):
            configs.append(triton.Config({"BLOCK_SIZE":block_size}, num_warps=warps, num_stages=num_stages))


@triton.autotune(configs=configs, key={"n_elements"})
@triton.jit
def seeded_dropout_kernel(x_ptr,
                   y_ptr,
                   n_elements,
                   p, seed,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    dropout_mask = tl.rand(seed, offsets) < p


    output = tl.where(dropout_mask, 0.0, x / (1-p))
    tl.store(y_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x, device=x.device, dtype=torch.float32)
    assert x.is_contiguous() and x.ndim == 2
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    seeded_dropout_kernel[grid](x, output, n_elements, p, seed)

    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=[128 * i for i in range(2, 20)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        plot_name="dropout-performance",
        args={"p": 0.2, "seed": 42},
        ylabel="GB/s",
        styles=[("green", "-"), ("blue", "-")],
    )
)
def benchmark(M: int, N: int, p: float, seed: int, provider: str) -> tuple[float, ...]:
    a = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.dropout(a, p, False), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: seeded_dropout(a, p, seed), quantiles=quantiles)

    gbps = lambda ms: a.numel() * a.element_size() / 1e9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)

def main() -> None:
    print("using device", DEVICE)
    a = torch.randn(5, 6, device=DEVICE, dtype=torch.float32)
    b = seeded_dropout(a, 0.5, 42)
    print(a)
    print(b)
    benchmark.run(show_plots=False, print_data=True, save_path=".")


if __name__ == "__main__":
    main()