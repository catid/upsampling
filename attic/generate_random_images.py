import os
import argparse
from tqdm.contrib.concurrent import process_map
from functools import partial
from PIL import Image
import numpy as np

def generate_random_image(output_dir, color_range, image_size, image_id):
    img_data = np.random.randint(color_range[0], color_range[1] + 1, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_data, mode="RGB")
    img.save(os.path.join(output_dir, f"random_image_{image_id:04d}.png"))

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Partially apply the additional arguments to process_image
    generate_random_image_with_args = partial(generate_random_image, args.output_dir, args.color_range, args.image_size)

    # Use process_map to handle the progress bar and processing
    process_map(generate_random_image_with_args, range(args.num_images), chunksize=2, desc="Processing images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random images.")
    parser.add_argument("--output_dir", default="random_images", help="Output directory for the generated images.")
    parser.add_argument("--num_images", type=int, default=20000, help="Number of images to generate.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Size of the generated images (width, height).")
    parser.add_argument("--color_range", type=int, nargs=2, default=[0, 255], help="Color range for the random pixels (min, max).")
    
    args = parser.parse_args()
    main(args)
