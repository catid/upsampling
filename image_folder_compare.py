import os
import lpips
import torch
import argparse
import numpy as np
from skimage import io, img_as_float32
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

def compute_metrics(input_folder, output_folder):
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net='alex')

    input_images = sorted(os.listdir(input_folder))
    output_images = sorted(os.listdir(output_folder))

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for inp_img_name, out_img_name in zip(input_images, output_images):
        input_image_path = os.path.join(input_folder, inp_img_name)
        output_image_path = os.path.join(output_folder, out_img_name)

        # Read the images
        input_image = img_as_float32(io.imread(input_image_path))
        output_image = img_as_float32(io.imread(output_image_path))

        # If images are RGBA, remove alpha channel
        if input_image.shape[2] == 4:
            input_image = input_image[:, :, :3]
        if output_image.shape[2] == 4:
            output_image = output_image[:, :, :3]

        # Compute PSNR and SSIM
        psnr = peak_signal_noise_ratio(input_image, output_image, data_range=1)
        ssim = structural_similarity(input_image, output_image, multichannel=True, channel_axis=-1, data_range=1)

        # Compute LPIPS
        output_image_eval = output_image * 2.0 - 1.0
        output_image_eval = np.clip(output_image_eval, -1, 1)
        input_image_eval = input_image * 2.0 - 1.0
        input_image_eval = np.clip(input_image_eval, -1, 1)
        lpips_score = calculate_lpips(output_image_eval, input_image_eval)

        print(f'{input_image_path}: PSNR={psnr} SSIM={ssim} LPIPS={lpips_score}')

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_score)

    return np.mean(psnr_values), np.mean(ssim_values), np.mean(lpips_values)


def main():
    parser = argparse.ArgumentParser(description='Compute PSNR, SSIM and LPIPS between two folders of images')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')

    args = parser.parse_args()

    mean_psnr, mean_ssim, mean_lpips = compute_metrics(args.input_folder, args.output_folder)

    print(f"Mean PSNR: {mean_psnr} SSIM: {mean_ssim} LPIPS: {mean_lpips}")


if __name__ == "__main__":
    main()

