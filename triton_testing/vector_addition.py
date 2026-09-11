import torch

import triton
import triton.language as tl

dv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(dv)

DEVICE = triton.runtime.driver.get_active_torch_device()

# 1) @triton.jit above function
# 2) takes pointers to x, y, and output, no return
# 3) takes n_elements
# 4) BLOCK_SIZE

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr):

    # seems like there are multiple programs, each deals with its own data
    pid = tl.program_id(axis=0) # axis=0 bc 1d launch grid

    

