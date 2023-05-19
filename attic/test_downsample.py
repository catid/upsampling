import time

import einops

import numpy as np

import torch
import torch.nn.functional as F


# Normal RGB image layout
rgb = np.random.randn(8, 8, 3)



# Use einops and then expand with numpy

t0 = time.time()

for i in range(1000):
    # Convert to correctly shaped tensor on CPU
    temp = einops.rearrange(rgb, '(h p1) (w p2) c -> (c p1 p2) h w', p1=2, p2=2)
    temp = np.expand_dims(temp, axis=0)

t1 = time.time()

output1 = torch.from_numpy(temp)

print(f"Option 1: einops first, then expand with numpy: shape={output1.shape} time={(t1 - t0)*1000.0:.3f} milliseconds")



# Expand with numpy and then use einops

t0 = time.time()

for i in range(1000):
    # Convert to correctly shaped tensor on CPU
    temp = np.expand_dims(rgb, axis=0)
    temp = einops.rearrange(temp, 'b (h p1) (w p2) c -> b (c p1 p2) h w', p1=2, p2=2)

t1 = time.time()

output2 = torch.from_numpy(temp)

print(f"Option 2: expand with numpy first, then einops : shape={output2.shape} time={(t1 - t0)*1000.0:.3f} milliseconds")




temp0 = torch.from_numpy(rgb)

t1 = time.time()

for i in range(1000):
    # Convert to correctly shaped tensor on GPU-compatible code
    temp = temp0.permute(2, 0, 1).unsqueeze(0)
    output3 = F.pixel_unshuffle(temp, downscale_factor=2)

t2 = time.time()

print(f"Option 3: torch permute, unsqueeze, then unshuffle: shape={output3.shape} time={(t1 - t0)*1000.0:.3f} milliseconds")




identical = torch.allclose(output1, output2)
print(f"1<->2 tensors are identical?  {identical}")

identical = torch.allclose(output2, output3)
print(f"2<->3 tensors are identical?  {identical}")

#(upsampling) ➜  upsampling git:(tiny) ✗ python attic/test_downsample.py
#Option 1: einops first, then expand with numpy: shape=torch.Size([1, 12, 4, 4]) time=3.883 milliseconds
#Option 2: expand with numpy first, then einops : shape=torch.Size([1, 12, 4, 4]) time=3.779 milliseconds
#Option 3: torch permute, unsqueeze, then unshuffle: shape=torch.Size([1, 12, 4, 4]) time=3.789 milliseconds
#1<->2 tensors are identical?  True
#2<->3 tensors are identical?  True
