import os
import struct
import argparse

from PIL import Image
import numpy as np
from tqdm.contrib.concurrent import process_map
from functools import partial

from tools.logging_tools import setup_colored_logging
import logging
setup_colored_logging()


def check_file(args, file_path):
    try:
        if args.fast and file_path.lower().endswith('.png'):
            # Do quick header check
            with open(file_path, 'rb') as f:
                # Read the first 24 bytes of the file
                header = f.read(24)

                if len(header) != 24:
                    logging.error(f"File {file_path} is truncated.")
                    return False

                # Verify the PNG signature
                png_signature = b'\x89PNG\r\n\x1a\n'
                if header[:8] != png_signature:
                    logging.error(f"File {file_path} does not have a valid PNG signature.")
                    return False

                # Unpack the width and height of the image from the header bytes
                unpacked = struct.unpack('>LL', header[16:24])
                img_width, img_height = unpacked

                # Check if the dimensions match the target dimensions
                if img_width < args.width or img_height < args.height:
                    logging.error(f"File {file_path} has the wrong dimensions.  Expected >= {args.width}x{args.height}  Found: {img_width}x{img_height}")
                    return False
                if img_width > args.max_width or img_height > args.max_height:
                    logging.error(f"File {file_path} has the wrong dimensions.  Expected < {args.max_width}x{args.max_height}  Found: {img_width}x{img_height}")
                    return False

        if args.fast and (file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg')):
            # Do quick header check
            with open(file_path, "rb") as f:
                # Read the first two bytes and check if they match the JPEG SOI marker
                soi_marker = f.read(2)
                if soi_marker != b'\xFF\xD8':
                    logging.error(f"File {file_path} is not a JPEG file.")
                    return False

                # Iterate through the segments to find the SOF0 marker (0xC0)
                while True:
                    # Find marker
                    marker = f.read(1)
                    if marker != b'\xff':
                        logging.error(f"File {file_path} is malformed marker={marker}.")
                        return False

                    while marker == b'\xff':
                        marker = f.read(1)

                    if marker == b'':
                        logging.error(f"File {file_path} is truncated.")
                        return False

                    length = struct.unpack(">H", f.read(2))[0]

                    # Break if reached the SOF0 marker
                    if marker == b'\xC0':
                        break

                    if length > 2:
                        # Move the file cursor to the next segment
                        f.seek(length - 2, 1)

                # Read the image dimensions
                dimensions_data = f.read(5)
                if len(dimensions_data) < 5:
                    logging.error(f"File {file_path} is truncated.")

                _, img_height, img_width = struct.unpack(">BHH", dimensions_data)

                # Check if the dimensions match the target dimensions
                if img_width < args.width or img_height < args.height:
                    logging.error(f"File {file_path} has the wrong dimensions.  Expected >= {args.width}x{args.height}  Found: {img_width}x{img_height}")
                    return False
                if img_width > args.max_width or img_height > args.max_height:
                    logging.error(f"File {file_path} has the wrong dimensions.  Expected < {args.max_width}x{args.max_height}  Found: {img_width}x{img_height}")
                    return False

        else:
            # Decode entire file to check
            img = Image.open(file_path)
            img_array = np.asarray(img)
            img_width, img_height = img.size

            # Check if the dimensions match the target dimensions
            if img_width < args.width or img_height < args.height:
                logging.error(f"File {file_path} has the wrong dimensions.  Expected >= {args.width}x{args.height}  Found: {img_width}x{img_height}")
                return False

    except Exception as e:
        logging.error(f"Error while reading {file_path}: {e}")
        return False

    return True

def process_file(args, file_path):
    if check_file(args, file_path):
        return True

    if args.clean:
        os.remove(file_path)
        logging.warning(f"Deleted: {file_path}")

    return False

def check_images(args):
    # Collect all files in root_directory and its subdirectories
    logging.info(f"Scanning for files in {args.data_dir} directory...")
    file_paths = []
    total_bytes = 0
    for dirpath, _, filenames in os.walk(args.data_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_bytes += os.path.getsize(file_path)
            file_paths.append(file_path)
    logging.info(f"Validating {len(file_paths)} files ({total_bytes / 1_000_000_000.0} GB)...")

    # Partially apply the additional arguments to process_image
    process_file_with_args = partial(process_file, args)

    # Use process_map to handle the progress bar and processing
    results = process_map(process_file_with_args, file_paths, chunksize=64, desc="Scanning")

    count = len(file_paths)
    success_count = sum(1 for x in results if x)

    return count, success_count


def all_entries_are_subdirectories(args):
    try:
        # List all the items in the directory
        items = os.listdir(args.data_dir)

        # Check if all items are subdirectories
        for item in items:
            item_path = os.path.join(args.data_dir, item)
            if not os.path.isdir(item_path):
                logging.error(f"Found normal file in root: {item_path}")

                if args.clean:
                    os.remove(item_path)
                    logging.warning(f"Deleted: {item_path}")
                else:
                    return False

    except Exception as e:
        print(f"Error: {e}")
        return False

    # If all items are subdirectories, return True
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate all images in the data directory")
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser("~/dataset"), help='Data directory to search for images')
    parser.add_argument('--width', type=int, default=256, help='Minimum width of the images.  Default: 256')
    parser.add_argument('--height', type=int, default=256, help='Minimum height of the images.  Default: 256')
    parser.add_argument('--max-width', type=int, default=2048, help='Minimum width of the images.  Default: 2048')
    parser.add_argument('--max-height', type=int, default=2048, help='Minimum height of the images.  Default: 2048')
    parser.add_argument('--clean', action='store_true', help='Automatically delete files that are invalid.')
    parser.add_argument('--fast', action='store_true', help='Just check the header to make sure the files are the right size and not truncated.')

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logging.warning(f"Data directory not found: {args.data_dir}")
        return

    if not all_entries_are_subdirectories(args):
        logging.warning(f"Found normal files within the data root.  Only subdirectories are allowed under {args.data_dir}")
        return

    count, success = check_images(args)
    failed = count - success

    if failed:
        logging.warning(f"Found {failed} files that were not valid images in the folder.  See above for details.")
        logging.info(f"Found {success} images with dimensions >= {args.width}x{args.height}")
    else:
        logging.info(f"Success: All {success} images in the dataset have dimensions >= {args.width}x{args.height}")

if __name__ == '__main__':
    main()
