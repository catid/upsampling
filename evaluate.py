import tools.normalization_factors

import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.ndimage import gaussian_filter

import lpips
from torchvision.transforms import ToTensor, Normalize, Compose

import argparse

from upsampling_net import create_vapsr2x

import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

def load_model(model_path, fp16):
    model = create_vapsr2x(rgb8output=True)
    model.load_state_dict(torch.load(model_path))
    if fp16:
        model.half()
    model.eval()

    #for name, param in model.named_parameters():
    #    logging.info(f"Name: {name}, Type: {param.dtype}, Size: {param.size()}")

    return model

# Define the function that calculates the LPIPS score
def calculate_lpips(img1, img2, fp16=True, use_gpu=True, net="alex", version="0.1"):
    # Transform the images into tensors
    img1_tensor = lpips.im2tensor(img1)
    img2_tensor = lpips.im2tensor(img2)

    if fp16:
        img1_tensor = img1_tensor.half()
        img2_tensor = img2_tensor.half()

    # Enable GPU if available and requested
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # Initialize the LPIPS model
    lpips_model = lpips.LPIPS(net=net, version=version, verbose=False)
    lpips_model.to(device)
    lpips_model.eval()

    # Calculate LPIPS
    with torch.no_grad():
        lpips_score = lpips_model(img1_tensor, img2_tensor).item()

    return lpips_score

def save_side_by_side(images, output_filename):
    # Create the output directory if it doesn't exist
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert numpy arrays to PIL Image objects
    pil_images = [Image.fromarray(np.uint8(img)) for img in images]

    # Get the dimensions of the combined image
    width, height = pil_images[0].size
    total_width = width * len(images)

    # Create a new PIL Image object for the combined image
    combined_image = Image.new("RGB", (total_width, height))

    # Paste images side-by-side into the combined image
    for i, img in enumerate(pil_images):
        combined_image.paste(img, (i * width, 0))

    # Save the combined image as a .png file
    output_path = os.path.join(output_dir, output_filename)
    combined_image.save(output_path)

# create a lower-resolution version of an image while mitigating potential
# aliasing artifacts caused by high-frequency components in the original image
def gaussian_downsample_image(image, decimation_factor=2):
    # Apply Gaussian blur
    blurred_image = gaussian_filter(image, sigma=decimation_factor / 6)

    # Decimate the image
    downsampled_image = blurred_image[::decimation_factor, ::decimation_factor]

    return downsampled_image

def evaluate(model, image_directory, crop_border=4, fp16=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    bicubic_psnr_values = []
    bicubic_ssim_values = []
    bicubic_lpips_values = []

    for img_name in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_name)

        #logging.info(f"Evaluating {img_path}")

        # Load the image and convert it to a tensor
        input_image = Image.open(img_path).convert("RGB")

        if args.gaussian:
            downsampled_image = Image.fromarray(gaussian_downsample_image(np.array(input_image), decimation_factor=2))
        else:
            downsampled_image = input_image.resize((input_image.width // 2, input_image.height // 2), Image.BICUBIC)

        bicubic_image = downsampled_image.resize((input_image.width, input_image.height), Image.BICUBIC)

        # Convert from 8-bit RGB image to tensor
        # We do layout change from NHWC to NCHW inside the model.
        # We also do normalization inside the model.
        downsampled_image = np.array(downsampled_image)
        input_tensor = torch.from_numpy(downsampled_image).unsqueeze(0).to(device)

        #print(f"input_tensor.shape = {input_tensor.shape}")
        #print(f"input_tensor.dtype = {input_tensor.dtype}")

        # Upsample the image using the model
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert the tensors back to 8-bit RGB images
        output_image_np = output_tensor.squeeze(0).cpu().numpy()

        #print(f"output_image_np.shape = {output_image_np.shape}")
        #print(f"output_image_np.dtype = {output_image_np.dtype}")


        # Rescale to -1..1 range
        output_image_np = output_image_np / 255.0
        output_image_np = np.clip(output_image_np, 0, 1)

        input_image_np = np.array(input_image) / 255.0
        input_image_np = np.clip(input_image_np, 0, 1)

        bicubic_image_np = np.array(bicubic_image) / 255.0
        bicubic_image_np = np.clip(bicubic_image_np, 0, 1)

        if crop_border != 0:
            input_image_np = input_image_np[crop_border:-crop_border, crop_border:-crop_border, ...]
            output_image_np = output_image_np[crop_border:-crop_border, crop_border:-crop_border, ...]
            bicubic_image_np = bicubic_image_np[crop_border:-crop_border, crop_border:-crop_border, ...]

        # Calculate PSNR and SSIM with RGB input ranging from 0..1
        psnr_value = psnr(input_image_np, output_image_np, data_range=1)
        ssim_value = ssim(input_image_np, output_image_np, multichannel=True, channel_axis=-1, data_range=1)

        bicubic_psnr_value = psnr(input_image_np, bicubic_image_np, data_range=1)
        bicubic_ssim_value = ssim(input_image_np, bicubic_image_np, multichannel=True, channel_axis=-1, data_range=1)

        # Rescale RGB from 0..1 to -1..1 for LPIPS calculation
        output_image_eval = output_image_np * 2.0 - 1.0
        output_image_eval = np.clip(output_image_eval, -1, 1)
        bicubic_image_eval = bicubic_image_np * 2.0 - 1.0
        bicubic_image_eval = np.clip(bicubic_image_eval, -1, 1)
        input_image_eval = input_image_np * 2.0 - 1.0
        input_image_eval = np.clip(input_image_eval, -1, 1)

        lpips_value = calculate_lpips(input_image_eval, output_image_eval, fp16=fp16)
        bicubic_lpips_value = calculate_lpips(input_image_eval, bicubic_image_eval, fp16=fp16)

        logging.info(f"{img_path}: PSNR={psnr_value} SSIM={ssim_value} LPIPS={lpips_value}")

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)

        bicubic_psnr_values.append(bicubic_psnr_value)
        bicubic_ssim_values.append(bicubic_ssim_value)
        bicubic_lpips_values.append(bicubic_lpips_value)

        save_side_by_side([input_image_np * 255, output_image_np * 255, bicubic_image_np * 255, downsampled_image], "output_" + img_name)

    return np.mean(psnr_values), np.mean(ssim_values), np.mean(lpips_values), np.mean(bicubic_psnr_values), np.mean(bicubic_ssim_values), np.mean(bicubic_lpips_values)

def main(args):
    fp16 = not args.fp32
    logging.info(f"Loading as FP16: {fp16}")

    model = load_model(args.model, fp16)
    mean_psnr, mean_ssim, mean_lpips, mean_bicubic_psnr, mean_bicubic_ssim, mean_bicubic_lpips = evaluate(model, args.image_dir, fp16=fp16)

    logging.info(f"Model PSNR: {mean_psnr} - Bicubic PSNR: {mean_bicubic_psnr}")
    logging.info(f"Model SSIM: {mean_ssim} - Bicubic SSIM: {mean_bicubic_ssim}")
    logging.info(f"Model LPIPS: {mean_lpips} - Bicubic LPIPS: {mean_bicubic_lpips}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model", type=str, default="upsampling.pth", help="Path to the model file produced by export_trained_model.py")
    parser.add_argument("--image-dir", type=str, default="./urban100/", help="Path to evaluation images")
    parser.add_argument('--gaussian', action='store_true', help='Use Gaussian downsampling instead of bicubic downsampling')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 network instead of FP16')

    args = parser.parse_args()

    main(args)
