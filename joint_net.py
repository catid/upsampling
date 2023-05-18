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


def space_to_depth(x, block_size):
    N, C, H, W = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(N, C * block_size ** 2, H // block_size, W // block_size)

def depth_to_space(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

class tiny_sr2(nn.Module):
    def __init__(self, rgb8output=True, channels=32):
        super(tiny_sr2, self).__init__()

        self.input_convert = FromInputRGB8()
        self.output_convert = ToOutputRGB(rgb8output=rgb8output)

        # 12 = 3 (RGB) * 2 (upscale factor) * 2 (space-to-depth)
        self.ds = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.d2s = depth_to_space(channels, 3, upscale_factor=4)

    def forward(self, rgb):
        rgb = self.input_convert(rgb)

        # Downsample the image by a factor of 2, converting to feature channels (without any convolution yet)
        rgb_sd = space_to_depth(rgb, 2)

        # Apply convolution to convert to ~32 feature channels
        feat = self.ds(rgb_sd)

        # Apply residual block(s)
        # FIXME: Try using torch.dot(a, b) instead of adding
        feat = self.rb(feat) + feat

        # Upsample the image by 4x to convert from features to RGB at twice original resolution
        out = self.d2s(feat)

        out = self.output_convert(out)

        return out


def create_joint2x(rgb8output):
    return tiny_sr2(
        channels=32,
        rgb8output=rgb8output)
