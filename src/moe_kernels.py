import torch
import os

os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl


@triton.jit
def _moe_forward_kernel(
    x_ptr,
    x_offsets_ptr,
    w1_ptr,
    w2_ptr,
    out_ptr,
    d_in,
    d_h,
    d_out,
    T,
    w1_ptr_stride_d_in,
    w1_ptr_stride_d_h,
    w2_ptr_stride_d_h,
    w2_ptr_stride_d_out,
    BLOCK_SIZE: tl.constexpr,
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


    """

    pid = tl.program_id(axis=0)

    group_id = pid // (GROUP_SIZE_T * GROUP_SIZE_DH)

    groups_along_d_out = tl.cdiv(d_out, GROUP_SIZE_DH)

    group_x = group_id % groups_along_d_out
    group_y = group_id // groups_along_d_out

    first_pid_in_group = group_id * GROUP_SIZE_T * GROUP_SIZE_DH

    group_relative_offset = pid - first_pid_in_group

    group_relative_m = group_relative_offset // GROUP_SIZE_DH
    group_relative_n = group_relative_offset % GROUP_SIZE_DH

    pid_m = group_y * GROUP_SIZE_T + group_relative_m
    pid_n = group_x * GROUP_SIZE_DH + group_relative_n

    print(f"Pid {pid}: ({pid_m}, {pid_n})")


def _moe_expert_forward(
    x: torch.Tensor, x_offsets: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)

    grid = lambda meta: (
        triton.cdiv(x.size(0), meta["GROUP_SIZE_T"])
        * triton.cdiv(w1.size(1), meta["GROUP_SIZE_DH"]),
    )

    BLOCK_SIZE = 64
    GROUP_SIZE_T = 2
    GROUP_SIZE_DH = 2
    GROUP_SIZE_DO = 2
    _moe_forward_kernel[grid](
        x_ptr=x,
        x_offsets_ptr=x_offsets,
        w1_ptr=w1,
        w2_ptr=w2,
        out_ptr=out,
        d_in=w1.size(0),
        d_h=w1.size(1),
        d_out=w2.size(1),
        T=x.size(0),
        w1_ptr_stride_d_in=w1.stride(0),
        w1_ptr_stride_d_h=w1.stride(1),
        w2_ptr_stride_d_h=w2.stride(0),
        w2_ptr_stride_d_out=w2.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_T=GROUP_SIZE_T,
        GROUP_SIZE_DH=GROUP_SIZE_DH,
        GROUP_SIZE_DO=GROUP_SIZE_DO,
    )

    return out


def main() -> None:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, T, C, d_h = 1, 2, 4, 6
    x = torch.randn((T * C,), device=DEVICE, dtype=torch.float32)
    x_offsets = torch.randint(
        0, 64, (T, C), device=DEVICE, dtype=torch.float32
    )  # need to figure this part out

    w1 = torch.randn(C, d_h, device=DEVICE, dtype=torch.float32)
    w2 = torch.randn(d_h, C, device=DEVICE, dtype=torch.float32)

    out = _moe_expert_forward(x, x_offsets, w1, w2)

    print(out)


if __name__ == "__main__":
    main()
