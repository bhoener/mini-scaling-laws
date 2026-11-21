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
    d,
    F,
    BT,
    x_stride_BT,
    x_stride_d,
    w1_stride_d,
    w1_stride_F,
    w2_stride_F,
    w2_stride_d,
    out_stride_BT,
    out_stride_d,
    BLOCK_SIZE_BT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
    GROUP_SIZE_T: tl.constexpr,
    GROUP_SIZE_F: tl.constexpr,
):
    """
      this is much simpler than i thought. we are just doing grouped_mm twice,
      instead of doing gate projections and rearranging tokens.

      what we need to do:
                          F
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

      also we are dealing with 3-dimensional tensors
      so we have to parallelize along B

      important not to forget that this is block matmul

      remember that our tensor is of shape (num_experts, j1, d_model)
      and the values are (BT, d_model)

      ok, if we assume d <= BLOCK_SIZE_D, things are simpler

      we can also choose not to parallelize along F and have each program loop locally

      accum = zeros
      for i in range(tl.cdiv(F, BLOCK_SIZE_F)):
        h = matmul(...)
        out = h @ W2[...]
        accum += out


        
                        [ x ]
        so              [ x ]
          d  [ x  x  x ][ x ]
   PID0 [ x ][ y  y  y ][ z ]
   PID1 [ x ][ y  y  y ][ z ]
   PID2 [ x ][ y  y  y ][ z ]
    """

    pid = tl.program_id(axis=0)

    group_id = pid // (GROUP_SIZE_T * GROUP_SIZE_F)


    groups_along_F = tl.cdiv(F, BLOCK_SIZE_F * GROUP_SIZE_F)

    group_x = group_id % groups_along_F
    group_y = group_id // groups_along_F

    first_pid_in_group = group_id * GROUP_SIZE_T * GROUP_SIZE_F

    group_relative_offset = pid - first_pid_in_group

    group_relative_m = group_relative_offset // GROUP_SIZE_F
    group_relative_n = group_relative_offset % GROUP_SIZE_F

    pid_m = group_y * GROUP_SIZE_T + group_relative_m
    pid_n = group_x * GROUP_SIZE_F + group_relative_n

    print(f"Pid {pid}: ({pid_m}, {pid_n})")

    # (B_t, 1) * (1, B_d)
    x_offsets = (tl.arange(0, BLOCK_SIZE_BT)[:, None] + pid_m * BLOCK_SIZE_BT) * x_stride_BT + tl.arange(0, BLOCK_SIZE_D)[None, :] * x_stride_d
    x_mask = ((tl.arange(0, BLOCK_SIZE_BT)[:, None] + pid_m * BLOCK_SIZE_BT) < BT) & (tl.arange(0, BLOCK_SIZE_D)[None, :] < d )

    # (B_d, 1) * (1, B_dh)
    w1_offsets = tl.arange(0, BLOCK_SIZE_D)[:, None] * w1_stride_d + tl.arange(0, BLOCK_SIZE_F)[None, :] * w1_stride_F
    w1_mask = ((tl.arange(0, BLOCK_SIZE_D)[:, None]) < d) & ((tl.arange(0, BLOCK_SIZE_F)[None, :]) < F)
    w2_offsets = (tl.arange(0, BLOCK_SIZE_F)[:, None]) * w2_stride_F + (tl.arange(0, BLOCK_SIZE_D)[None, :]) * w2_stride_d
    w2_mask = ((tl.arange(0, BLOCK_SIZE_F)[:, None]) < F) & ((tl.arange(0, BLOCK_SIZE_D)[None, :]) < d)


    x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_D), dtype=tl.float32)

    #print(x_offsets.shape, x_mask.shape)
    
    #print(f"pid {pid} | i={i} |", x_offsets.shape, x_mask.shape)

    for i in range(tl.cdiv(F, BLOCK_SIZE_F)):
        w1_block = tl.load(w1_ptr + w1_offsets, mask=w1_mask, other=0.0)
        w2_block = tl.load(w2_ptr + w2_offsets, mask=w2_mask, other=0.0)

        print("w1_offsets", w1_offsets)
        #print("x_block:", x_block, "w1_block:", w1_block)

        h = tl.dot(x_block, w1_block)

    

        h_act = h * tl.sigmoid(h)

        accumulator = tl.dot(h_act, w2_block, acc=accumulator)

        w1_offsets += BLOCK_SIZE_F * w1_stride_F
        w2_offsets += BLOCK_SIZE_F * w2_stride_F

        # print(w1_offsets, w2_offsets)
        
        
        w1_mask = ((tl.arange(0, BLOCK_SIZE_D)[:, None]) < d) & ((tl.arange(0, BLOCK_SIZE_F)[None, :] + i * BLOCK_SIZE_F) < F)
        w2_mask = ((tl.arange(0, BLOCK_SIZE_F)[:, None] + i * BLOCK_SIZE_F) < F) & ((tl.arange(0, BLOCK_SIZE_D)[None, :]) < d)

        #print("h:", h, "h_act:", h_act)

    out_offsets_bt = tl.arange(0, BLOCK_SIZE_BT) + pid * BLOCK_SIZE_BT
    out_offsets_d = tl.arange(0, BLOCK_SIZE_D)

    out_mask = (out_offsets_bt[:, None] < BT) & (out_offsets_d[None, :] < d)

    out_offsets = out_offsets_bt[:, None] * out_stride_BT + out_offsets_d[None, :] * out_stride_d

    tl.store(out_ptr + out_offsets, accumulator, out_mask)


def _moe_expert_forward(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)

    grid = lambda meta: (
        triton.cdiv(x.size(0), meta["BLOCK_SIZE_BT"])
        * triton.cdiv(w1.size(1), meta["BLOCK_SIZE_D"]),
    )

    BLOCK_SIZE_BT = 4
    BLOCK_SIZE_D = triton.next_power_of_2(x.size(1))
    GROUP_SIZE_T = 2
    GROUP_SIZE_F = 2
    _moe_forward_kernel[grid](
        x_ptr=x,
        w1_ptr=w1,
        w2_ptr=w2,
        out_ptr=out,
        d=x.size(0),
        F=w1.size(1),
        BT=x.size(0),
        x_stride_BT=x.stride(0),
        x_stride_d=x.stride(1),
        w1_stride_d=w1.stride(0),
        w1_stride_F=w1.stride(1),
        w2_stride_F=w2.stride(0),
        w2_stride_d=w2.stride(1),
        out_stride_BT=out.stride(0),
        out_stride_d=out.stride(1),
        BLOCK_SIZE_BT=BLOCK_SIZE_BT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_F=BLOCK_SIZE_D,
        GROUP_SIZE_T=GROUP_SIZE_T,
        GROUP_SIZE_F=GROUP_SIZE_F,
    )

    return out


def main() -> None:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, T, C, F = 2, 16, 32, 128
    print(torch.zeros(triton.cdiv(T, 4), triton.cdiv(F, 4)))
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

    w1 = torch.randn(C, F, device=DEVICE, dtype=torch.float32) / C**0.5
    w2 = torch.randn(F, C, device=DEVICE, dtype=torch.float32) / F**0.5

    out = _moe_expert_forward(x, w1, w2)

    print(out)

    print("w1 numel:", w1.numel())

if __name__ == "__main__":
    main()
