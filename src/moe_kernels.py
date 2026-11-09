import torch
import os
torch.set_printoptions(precision=2, sci_mode=False)
os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl


@triton.jit
def _moe_forward_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    out_ptr,
    d_in,
    d_h,
    d_out,
    BT,
    x_stride_BT,
    x_stride_d_in,
    w1_stride_d_in,
    w1_stride_d_h,
    w2_stride_d_h,
    w2_stride_d_out,
    BLOCK_SIZE_BT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_DH: tl.constexpr,
    BLOCK_SIZE_DO: tl.constexpr,
    GROUP_SIZE_T: tl.constexpr,
    GROUP_SIZE_DH: tl.constexpr,
    GROUP_SIZE_DO: tl.constexpr,
):
    """
      this is much simpler than i thought. we are just doing grouped_mm twice,
      instead of doing gate projections and rearranging tokens.

      what we need to do:
                          d_h
                     [ x  x  x  x ]
                   d [ x  x  x  x ]
                     [ x  x  x  x ]
      [     d
    t1[[ x  x  x ]   [ 0  1 | 4  5]
       [ x  x  x ]], [ 2  3 | 6  7]
                     --------------
    t2[[ x  x  x ]], [ 8  9 |12 13]

    t3[[ x  x  x ]   [10 11 |14 15]
                     --------------
       [ x  x  x ]   [16 17 |20 21]
       [ x  x  x ]], [18 19 |22 23]
      ]

      can we just do one pid per expert?

      probably not. we should use groups

      note also that we need to do this twice.

      we also must read the offsets from the index tensors

      also we are dealing with 3-dimensional tensors
      so we have to parallelize along B

      important not to forget that this is block matmul

      remember that our tensor is of shape (num_experts, j1, d_model)
      and the values are (BT, d_model)


    """

    pid = tl.program_id(axis=0)

    group_id = pid // (GROUP_SIZE_T * GROUP_SIZE_DH)


    groups_along_d_h = tl.cdiv(d_h, BLOCK_SIZE_DH * GROUP_SIZE_DH)

    group_x = group_id % groups_along_d_h
    group_y = group_id // groups_along_d_h

    first_pid_in_group = group_id * GROUP_SIZE_T * GROUP_SIZE_DH

    group_relative_offset = pid - first_pid_in_group

    group_relative_m = group_relative_offset // GROUP_SIZE_DH
    group_relative_n = group_relative_offset % GROUP_SIZE_DH

    pid_m = group_y * GROUP_SIZE_T + group_relative_m
    pid_n = group_x * GROUP_SIZE_DH + group_relative_n

    print(f"Pid {pid}: ({pid_m}, {pid_n})")

    # (B_t, 1) * (1, B_d)
    x_offsets = (tl.arange(0, BLOCK_SIZE_BT)[:, None] + pid_m * BLOCK_SIZE_BT) * x_stride_BT + tl.arange(0, BLOCK_SIZE_D)[None, :] * x_stride_d_in 
    x_mask = ((tl.arange(0, BLOCK_SIZE_BT)[:, None] + pid_m * BLOCK_SIZE_BT) < BT) & (tl.arange(0, BLOCK_SIZE_D)[None, :] < d_in )

    # (B_d, 1) * (1, B_dh)
    w1_offsets = tl.arange(0, BLOCK_SIZE_D)[:, None] * w1_stride_d_in + (tl.arange(0, BLOCK_SIZE_DH)[None, :] + pid_n * BLOCK_SIZE_DH) * w1_stride_d_h
    w1_mask = (tl.arange(0, BLOCK_SIZE_D)[:, None] < d_in) & ((tl.arange(0, BLOCK_SIZE_DH)[None, :] + pid_n * BLOCK_SIZE_DH) < d_h)

    
    accumulator = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_DH), dtype=tl.float32)

    print(x_offsets.shape, x_mask.shape)
    for i in range(tl.cdiv(d_in, BLOCK_SIZE_D)):
        print(x_offsets.shape, x_mask.shape)
        x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        w1_block = tl.load(w1_ptr + w1_offsets, mask=w1_mask, other=0.0)

        accumulator = tl.dot(x_block, w1_block, acc=accumulator) # possibly just =

        x_offsets += BLOCK_SIZE_D * x_stride_d_in
        w1_offsets += BLOCK_SIZE_D * w1_stride_d_in

        x_mask = ((tl.arange(0, BLOCK_SIZE_D)[:, None] + i * BLOCK_SIZE_D) < d_in) & ((tl.arange(0, BLOCK_SIZE_BT)[None, :] + pid_m * BLOCK_SIZE_BT) < BT)
        w1_mask = ((tl.arange(0, BLOCK_SIZE_D)[None, :] + i * BLOCK_SIZE_D) < d_in) & ((tl.arange(0, BLOCK_SIZE_DH)[:, None] + pid_n * BLOCK_SIZE_DH) < d_h)

    print(accumulator)

def _moe_expert_forward(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)

    grid = lambda meta: (
        triton.cdiv(x.size(0), meta["BLOCK_SIZE_BT"])
        * triton.cdiv(w1.size(1), meta["BLOCK_SIZE_DH"]),
    )

    BLOCK_SIZE_BT = 4
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_DH = 4
    BLOCK_SIZE_DO = 4,
    GROUP_SIZE_T = 2
    GROUP_SIZE_DH = 2
    GROUP_SIZE_DO = 2
    _moe_forward_kernel[grid](
        x_ptr=x,
        w1_ptr=w1,
        w2_ptr=w2,
        out_ptr=out,
        d_in=x.size(0),
        d_h=w1.size(1),
        d_out=x.size(0),
        BT=x.size(0),
        x_stride_BT=x.stride(0),
        x_stride_d_in=x.stride(1),
        w1_stride_d_in=w1.stride(0),
        w1_stride_d_h=w1.stride(1),
        w2_stride_d_h=w2.stride(0),
        w2_stride_d_out=w2.stride(1),
        BLOCK_SIZE_BT=BLOCK_SIZE_BT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_DH=BLOCK_SIZE_DH,
        BLOCK_SIZE_DO=BLOCK_SIZE_DO,
        GROUP_SIZE_T=GROUP_SIZE_T,
        GROUP_SIZE_DH=GROUP_SIZE_DH,
        GROUP_SIZE_DO=GROUP_SIZE_DO,
    )

    return out


def main() -> None:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, T, C, d_h = 2, 16, 32, 48
    print(torch.zeros(triton.cdiv(T, 4), triton.cdiv(d_h, 4)))
    x = torch.randn((B*T, C), device=DEVICE, dtype=torch.float32)

    test = [
        [[1, 2, 3, 4]],

        [[5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]],

         [[17, 18, 19, 20],
          [21, 22, 23, 24]]
    ]

    test_tensor = torch.nested.nested_tensor(test, layout=torch.jagged)
    print(test_tensor)
    print(test_tensor.values())

    w1 = torch.randn(C, d_h, device=DEVICE, dtype=torch.float32)
    w2 = torch.randn(d_h, C, device=DEVICE, dtype=torch.float32)

    out = _moe_expert_forward(x, w1, w2)

    print(out)


if __name__ == "__main__":
    main()
