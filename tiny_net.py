import torch
import torch.nn as nn
import torch.nn.functional as F




class FromInputRGB8(nn.Module):
    def __init__(self):
        super(FromInputRGB8, self).__init__()

    def forward(self, x):
        # Convert data type to FP16 and normalize from [0, 255] to [-1, 1]
        x = x.half()
        x = (x / 127.5) - 1.0

        return x

class ToOutputRGB(nn.Module):
    def __init__(self, rgb8output):
        super(ToOutputRGB, self).__init__()

        self.rgb8output = rgb8output

    def forward(self, x):
        if self.rgb8output:
            # Scale from [-1, 1] to [0, 255], rounding to nearest integer
            # Note: Combined addition by 127.5 with adding 0.5 round-up factor
            x = x * 127.5 + 128
            x = torch.clamp(x, 0, 255)
            x = x.byte()
        else:
            x = torch.clamp(x, -1, 1)

        return x

def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)

class SRB(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels), # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False), # Pointwise convolution
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels), # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False), # Pointwise convolution
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        shorcut = x.clone()
        x = self.rb(x)
        x = x + shorcut
        return x

class tiny2x(nn.Module):
    def __init__(self, d2sinput=True, rgb8output=True, channels=48, blocks=1):
        super(tiny2x, self).__init__()

        self.input_convert = FromInputRGB8()
        self.output_convert = ToOutputRGB(rgb8output=rgb8output)

        self.d2sinput = d2sinput
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2 * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.body = make_layer(SRB, blocks, channels)

        # Upsample the image by 4x to convert from features to RGB at twice original resolution
        self.d2s = nn.Sequential(
            #nn.Conv2d(channels, 64, 3, 1, 1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels), # Depthwise convolution
            nn.Conv2d(channels, 64, kernel_size=1, stride=1, padding=0, bias=False), # Pointwise convolution
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            #nn.Conv2d(16, 3 * 4, 3, 1, 1, bias=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False, groups=16), # Depthwise convolution
            nn.Conv2d(16, 3 * 4, kernel_size=1, stride=1, padding=0, bias=False), # Pointwise convolution
            nn.PixelShuffle(2),
        )

    def forward(self, rgb):
        # If d2sinput is False, then the input shape should be BCHW, where C=3
        # If d2sinput is True, then the input shape should be BCHW, where C=12
        if not self.d2sinput:
            rgb = F.pixel_unshuffle(rgb, downscale_factor=2)

        feat = self.input_convert(rgb)

        feat = self.conv1(feat)

        # Apply residual blocks
        feat = self.body(feat)

        # Upsample the image by 4x to convert from features to RGB at twice original resolution
        out = self.d2s(feat)

        out = self.output_convert(out)

        return out

# Network expects BCHW input shape
# d2sinput: Input channels are depth-to-space converted from 3 to 12 channels by caller
# rgb8output: Output channels are converted to 8-bit inside the network
# Inference mode should set both to True for performance.
# Training mode should set both to False.
def create_tiny2x(d2sinput, rgb8output):
    return tiny2x(
        channels=32,
        d2sinput=d2sinput,
        rgb8output=rgb8output)
