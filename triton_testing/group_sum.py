import torch
import triton
import triton.language as tl

"""
This is probably slower than just summing a tensor directly along columns. However,
since we are making a layernorm kernel, it is useful to have to avoid writing the entire tensor to memory
before summing and fuse operations together
"""

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@triton.jit
def _group_sum_kernel_1(in_ptr, intermediate_ptr, locks_ptr, M, N, GROUP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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
    offsets = tl.arange(0, BLOCK_SIZE)

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
    intermediate_offsets = tl.arange(0, BLOCK_SIZE)
    mask_intermediate = intermediate_offsets < N

    # keep checking that the lock pointer is zero
    # if it is, replace it with one and move on
    while tl.atomic_cas(pid_lock_ptr, 0, 1) == 1:
        pass

    # check if we are the first pid to access
    count = tl.load(pid_lock_accessed_ptr)
    if count == 0:
        # if so, we don't need to load the whole intermediate row
        tl.atomic_xchg(pid_lock_accessed_ptr, 1)
    else:
        # otherwise, load and add our contribution
        row_contribution += tl.load(intermediate_block_start + intermediate_offsets, mask=mask_intermediate, other=0.0)

    # store back to intermediate
    tl.store(intermediate_block_start + intermediate_offsets, row_contribution, mask=mask_intermediate)

    # unlock
    tl.atomic_xchg(pid_lock_ptr, 0)


@triton.jit
def _group_sum_kernel_2(intermediate_ptr, out_ptr, GROUP_SIZE: tl.constexpr, N, BLOCK_SIZE: tl.constexpr):
    """
    Now we just sum along cols

    [ |  |  |  | ]
    [ |  |  |  | ]
    [ |  |  |  | ]
    [ V  V  V  V ] => [ x  x  x  x ]

    we make the assumption that M < BLOCK_SIZE
    """

    # the column we get corresponds to our pid
    col = tl.program_id(axis=0)

    # assume our intermediate tensor is contiguous and get offsets for the col
    offsets = tl.arange(0, BLOCK_SIZE) * N + col

    # mask for loading
    mask = tl.arange(0, BLOCK_SIZE) < GROUP_SIZE

    # load and sum
    col_data = tl.load(intermediate_ptr + offsets, mask=mask, other=0.0)
    sum_res = tl.sum(col_data, axis=0)

    # store result into output ptr
    tl.store(out_ptr + col, sum_res)





def group_sum(x: torch.Tensor, GROUP_SIZE: int = 256, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous
    x = x.to(torch.float32)
    x = x.to(DEVICE)
    M, N = x.size()
    assert BLOCK_SIZE >= M and BLOCK_SIZE >= N

    # Apparently @triton.autotune breaks things?
    # Why is this? 
    # hmm
    # i think it runs the kernel on the given pointers multiple times
    # so we cannot assume that locks and inter are all zeros to begin with
    # running the tests messes things up
    # will just do manual for now

    # create zeroed-out intermediate tensor
    inter = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=DEVICE)

    # create locks tensor
    locks = torch.zeros(GROUP_SIZE * 2, dtype=torch.int32, device=DEVICE)

    # populate intermediate tensor
    _group_sum_kernel_1[(M,)](x, inter, locks, M, N, GROUP_SIZE, BLOCK_SIZE)

    # create empty output
    out = torch.empty(N, device=DEVICE, dtype=torch.float32)

    # populate output
    _group_sum_kernel_2[(N,)](inter, out, GROUP_SIZE, N, BLOCK_SIZE)

    return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=[128 * i for i in range(1, 9)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        plot_name="group-sum-performance",
        args={"GROUP_SIZE": 256, "BLOCK_SIZE": 1024},
        xlabel="size",
        ylabel="GB/s",
        styles=[("green", "-"), ("blue", "-")],
    )
)
def benchmark(M: int, N: int, provider: str, GROUP_SIZE: int=256, BLOCK_SIZE: int=256) -> tuple[int, ...]:
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x.sum(0), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: group_sum(x), quantiles=quantiles)

    gbps = lambda ms: x.numel() * x.element_size() / 1e9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)

if __name__ == "__main__":
    x = torch.randn(24, 32, device=DEVICE, dtype=torch.float32)

    x_summed = group_sum(x)

    x_summed_ref = x.sum(dim=0)

    print(x_summed)
    print(x_summed_ref)

    torch.testing.assert_close(x_summed, x_summed_ref, atol=1e-2, rtol=0)
    
    print("PASSED")

    # BRO WHY DO WE MOG SO HARD WHAT IS THIS 25% SPEEDUP
    # nvm torch wins at 1024x1024
    benchmark.run(show_plots=False, print_data=True, save_path=".")


