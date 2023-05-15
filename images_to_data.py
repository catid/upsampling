import argparse
import os
import glob
from PIL import Image
from pathlib import Path
import multiprocessing

from tqdm.contrib.concurrent import process_map
from functools import partial

from tools.dataset_tools import save_random_image_crops_to_disk

from tools.logging_tools import setup_colored_logging
import logging
setup_colored_logging()


def process_image(args, input_file):
    try:
        img = Image.open(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        if img.format == 'PNG':
            # PNG images do not subsample chroma
            downsample_first = False
        else:
            # But other formats do.  WebP doesn't
            downsample_first = True
        
        if args.no_downsample:
            # If you are using lossless Webp set the --no-downsample flag
            downsample_first = False

        save_random_image_crops_to_disk(
            img,
            base_name,
            args.output_dir,
            args.width,
            args.height,
            downsample_first=downsample_first)

    except Exception as e:
        logging.warning(f"Failed to process {input_file}: {e}")

def slice_images(args):
    image_files = glob.glob(f"{args.input_dir}/*.png") + glob.glob(f"{args.input_dir}/*.jpg") + glob.glob(f"{args.input_dir}/*.jpeg")
    os.makedirs(args.output_dir, exist_ok=True)

    # Partially apply the additional arguments to process_image
    process_image_with_args = partial(process_image, args)

    # Use process_map to handle the progress bar and processing
    process_map(process_image_with_args, image_files, chunksize=1, desc="Processing images")

def create_and_verify_subfolder_path(path):
    # Normalize the path
    path = os.path.normpath(path)
    path_parts = path.split(os.sep)

    # Verify that the provided path contains at least a folder and a subfolder
    if len(path_parts) < 2:
        raise ValueError("Invalid output path. The path must contain at least a folder and a subfolder.")

    # Check if the last part of the path has any file-like extension
    if os.path.splitext(path_parts[-1])[1]:
        raise ValueError("Invalid output path. The last part of the path should not contain any file-like extension.")

    # Create the folder and subfolder path without failing if the path exists
    os.makedirs(path, exist_ok=True)

    # Verify that no files exist in the path
    if any(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path)):
        raise ValueError("The provided output path contains files. It should be empty.")


def at_least_one_file_exists(folder_path):
    # List all the items in the folder
    items = os.listdir(folder_path)

    # Check if there is at least one file in the folder
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            return True

    # If no file was found, return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Slice images into smaller non-overlapping parts")
    parser.add_argument("input_dir", type=str, default="extracted_frames", help="Path to the directory containing input images")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the sliced images will be saved.  Should be a subfolder of data/ for DALI to load them properly.")
    parser.add_argument("--width", type=int, default=512, help="Crop width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Crop height (default: 512)")
    parser.add_argument('--no-downsample', action='store_true', help='Disable downsampling before cropping.  If the image is large enough it will still downsample repeatedly to enable learning multi-scale features')

    args = parser.parse_args()

    try:
        if not at_least_one_file_exists(args.input_dir):
            raise ValueError("The provided input path does not contain any files")

        create_and_verify_subfolder_path(args.output_dir)

        slice_images(args)

        logging.info(f"Operation successful")

    except Exception as e:
        logging.error(f"{e}")

if __name__ == "__main__":
    main()
