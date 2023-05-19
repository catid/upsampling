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

def depth_to_space(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * upscale_factor, 3, 1, 1, bias=False)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, relu, upconv2, pixel_shuffle])

def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)

class SRB(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, groups=channels), # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True), # Pointwise convolution
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, groups=channels), # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True), # Pointwise convolution
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        shorcut = x.clone()
        x = self.rb(x)
        x = x + shorcut
        return x

class tiny_sr2(nn.Module):
    def __init__(self, rgb8output=True, channels=32, blocks=2):
        super(tiny_sr2, self).__init__()

        self.input_convert = FromInputRGB8()
        self.output_convert = ToOutputRGB(rgb8output=rgb8output)

        self.d2s_conv = nn.Sequential(
            nn.Conv2d(3 * 2 * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.body = make_layer(SRB, blocks, channels)

        self.d2s = depth_to_space(channels, 3, upscale_factor=4)

    def forward(self, rgb):
        rgb = self.input_convert(rgb)

        # Downsample 2x and convolve
        feat = F.pixel_unshuffle(rgb, downscale_factor=2)
        feat = self.d2s_conv(feat)

        # Apply residual blocks
        feat = self.body(feat)

        # Upsample the image by 4x to convert from features to RGB at twice original resolution
        out = self.d2s(feat)

        out = self.output_convert(out)

        return out


def create_joint2x(rgb8output):
    return tiny_sr2(
        channels=32,
        rgb8output=rgb8output)
