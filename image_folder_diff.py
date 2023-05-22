import os
import numpy as np
from PIL import Image
import argparse

def generate_difference(input_folder1, input_folder2, output_folder):
    input_images1 = sorted(os.listdir(input_folder1))
    input_images2 = sorted(os.listdir(input_folder2))

    for inp_img_name1, inp_img_name2 in zip(input_images1, input_images2):
        input_image_path1 = os.path.join(input_folder1, inp_img_name1)
        input_image_path2 = os.path.join(input_folder2, inp_img_name2)
        output_image_path = os.path.join(output_folder, inp_img_name1)

        # Open image files
        with Image.open(input_image_path1) as img1, Image.open(input_image_path2) as img2:
            # Convert images to numpy arrays
            img1_arr = np.array(img1)
            img2_arr = np.array(img2)
            
            # If images are RGBA, remove alpha channel
            if img1_arr.shape[2] == 4:
                img1_arr = img1_arr[:, :, :3]
            if img2_arr.shape[2] == 4:
                img2_arr = img2_arr[:, :, :3]

            # Compute the absolute difference
            diff = np.abs(img1_arr.astype(np.float32) - img2_arr.astype(np.float32))
            
            # Convert to 8-bit image
            diff = (diff / diff.max() * 255).astype(np.uint8)

            # Convert numpy array back to image and save
            Image.fromarray(diff).save(output_image_path)

def main():
    parser = argparse.ArgumentParser(description='Generate the difference between two folders of images')
    parser.add_argument('--input_folder1', type=str, required=True, help='First input folder path')
    parser.add_argument('--input_folder2', type=str, required=True, help='Second input folder path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')

    args = parser.parse_args()

    generate_difference(args.input_folder1, args.input_folder2, args.output_folder)

if __name__ == "__main__":
    main()

