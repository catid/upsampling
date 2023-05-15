



# PNG images store gamma-corrected sRGB values.

# This function converts from PNG gamma-corrected sRGB to linear sRGB (actual light intensity).
def ConvertFromRGBToLinear(x):
    return torch.where(x >= 0.04045, torch.pow((x + 0.055) / (1.0 + 0.055), 2.4), x / 12.92)

# This function converts from linear sRGB (actual light intensity) to PNG gamma-corrected sRGB.
def ConvertFromLinearToRGB(x):
    return torch.where(x >= 0.0031308, 1.055 * torch.pow(x, 1.0/2.4) - 0.055, x * 12.92)


# YUV for BT.709 standard: https://en.wikipedia.org/wiki/YUV

class RGB2YUV(nn.Module):
    def __init__(self, half=True):
        super(RGB2YUV, self).__init__()
        # BT.709 standard
        self.lsrgb2yuv_matrix = torch.tensor([[0.2126, -0.114572, 0.615],
                                            [0.7152, -0.385428, -0.51499],
                                            [0.0722, 0.5, -0.1]])
    
        if half:
            self.lsrgb2yuv_matrix = self.lsrgb2yuv_matrix.half()

    def forward(self, x):
        # Scale from -1..1 to 0..1
        x = (x + 1) / 2

        # PNG images store gamma-corrected sRGB values.
        # This function converts to linear sRGB, which is actual light intensity.
        lsrgb = ConvertFromRGBToLinear(x)

        # Convert from linear sRGB to YUV with BT.709 colorspace
        yuv = torch.einsum('ij,bcwh->bcwh', self.lsrgb2yuv_matrix.to(x.device), lsrgb)

        return yuv

class YUV2RGB(nn.Module):
    def __init__(self, half=True):
        super(YUV2RGB, self).__init__()
        self.yuv2lsrgb_matrix = torch.tensor([[1, 1, 1],
                                            [0, -0.187324, 1.8556],
                                            [1.5748, -0.468124, 0]])
        
        if half:
            self.yuv2lsrgb_matrix = self.yuv2lsrgb_matrix.half()

    def forward(self, x):
        # Convert from YUV with BT.709 colorspace to linear sRGB
        lsrgb = torch.einsum('ij,bcwh->bcwh', self.yuv2lsrgb_matrix.to(x.device), x)

        # PNG images store gamma-corrected sRGB values.
        # This function converts from linear sRGB (actual light intensity) to gamma-corrected sRGB.
        srgb = ConvertFromLinearToRGB(lsrgb)

        # Scale back to -1..1 range
        srgb = srgb * 2 - 1

        return srgb


# OKLAB color space: https://bottosson.github.io/posts/oklab/
# Linear sRGB: https://bottosson.github.io/posts/colorwrong/#what-can-we-do

class RGB2OKLAB(nn.Module):
    def __init__(self, half=True):
        super(RGB2OKLAB, self).__init__()

        self.lsrgb2lms_matrix = torch.tensor(
            [[0.4122214708, 0.5363325363, 0.0514459929],
             [0.2119034982, 0.6806995451, 0.1073969566],
             [0.0883024619, 0.2817188376, 0.6299787005]])

        self.lms2lab_matrix = torch.tensor(
            [[0.2104542553, 0.7936177850, -0.0040720468],
             [1.9779984951, -2.4285922050, 0.4505937099],
             [0.0259040371, 0.7827717662, -0.8086757660]])

        if half:
            self.lsrgb2lms_matrix = self.lsrgb2lms_matrix.half()
            self.lms2lab_matrix = self.lms2lab_matrix.half()

    def forward(self, x):
        # Rescale input from -1..1 to 0..1
        x = (x + 1) / 2

        # PNG images store gamma-corrected sRGB values.
        # This function converts to linear sRGB, which is actual light intensity.
        lsrgb = ConvertFromRGBToLinear(x)

        # Convert to LMS
        lms = torch.einsum('ij,bcwh->bcwh', self.lsrgb2lms_matrix.to(x.device), lsrgb)

        # Apply cube-root non-linearity from OKLAB algorithm
        lms = torch.pow(lms, 1/3)

        # Convert to OKLAB from LMS
        oklab = torch.einsum('ij,bcwh->bcwh', self.lms2lab_matrix.to(x.device), lms)

        return oklab

class OKLAB2RGB(nn.Module):
    def __init__(self, half=True):
        super(OKLAB2RGB, self).__init__()

        self.lms2lsrgb_matrix = torch.tensor(
            [[4.0767416621, -3.3077115913, 0.2309699292],
             [-1.2684380046, 2.6097574011, -0.3413193965],
             [-0.0041960863, -0.7034186147, 1.7076147010]]).half()
        
        if half:
            self.lms2lsrgb_matrix = self.lms2lsrgb_matrix.half()

    def forward(self, x):
        # Convert to LMS from OKLAB
        lms_ = torch.zeros_like(x)
        lms_[:, 0, :, :] = x[:, 0, :, :] + 0.3963377774 * x[:, 1, :, :] + 0.2158037573 * x[:, 2, :, :]
        lms_[:, 1, :, :] = x[:, 0, :, :] - 0.1055613458 * x[:, 1, :, :] - 0.0638541728 * x[:, 2, :, :]
        lms_[:, 2, :, :] = x[:, 0, :, :] - 0.0894841775 * x[:, 1, :, :] - 1.2914855480 * x[:, 2, :, :]

        lms = torch.pow(lms_, 3)

        # Convert to Linear sRGB from LMS
        lsrgb = torch.einsum('ij,bcwh->bcwh', self.lms2lsrgb_matrix.to(x.device), lms)

        # PNG images store gamma-corrected sRGB values.
        # This function converts from linear sRGB (actual light intensity) to gamma-corrected sRGB.
        srgb = ConvertFromLinearToRGB(lsrgb)

        # Rescale output from 0..1 to -1..1
        srgb = srgb * 2 - 1

        return srgb














