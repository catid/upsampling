import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm



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


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)





# From https://raw.githubusercontent.com/zhoumumu/VapSR/main/code/vapsr.py

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

class VAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 3, padding=1) # changed from 1 to 3
        self.activation = nn.GELU()
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x

def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

#both scale X2 and X3 use this version
def pixelshuffle_single(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])

class vapsr(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, d_atten=64, conv_groups=1, rgb8output=True):
        super(vapsr, self).__init__()

        self.input_convert = FromInputRGB8()
        self.output_convert = ToOutputRGB(rgb8output=rgb8output)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(VAB, num_block, num_feat, d_atten)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=conv_groups) #conv_groups=2 for VapSR-S

        # upsample
        if scale == 4:
            self.upsampler = pixelshuffle(num_feat, num_out_ch, upscale_factor=scale)
        else:
            self.upsampler = pixelshuffle_single(num_feat, num_out_ch, upscale_factor=scale)

    def forward(self, feat):
        feat = self.input_convert(feat)

        feat = self.conv_first(feat)
        body_feat = self.body(feat)
        body_out = self.conv_body(body_feat)
        feat = feat + body_out
        out = self.upsampler(feat)

        out = self.output_convert(out)

        return out




# In training mode produces FP16 -1..1 RGB output
# In eval mode produces UINT8 0..255 RGB output
def create_vapsr2x(rgb8output):
    return vapsr(
        num_in_ch=3,
        num_out_ch=3,
        scale=2,
        num_feat=64,
        num_block=20,
        d_atten=48,
        conv_groups=1,
        rgb8output=rgb8output)
