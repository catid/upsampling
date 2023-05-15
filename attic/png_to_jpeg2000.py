import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import numpy as np

from tqdm import tqdm
import glymur

def convert_png_to_jpeg2000(file, output_path):
    # Load the PNG image
    img = Image.open(file)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Create the output file path with .jp2 extension
    output_file = output_path / (file.stem + ".jp2")

    # Save the image as lossless JPEG2000
    #glymur.set_option('lib.num_threads', 2)
    glymur.Jp2k(str(output_file), data=img_array, cratios=[1])

def convert_to_jpeg2000(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create the output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        input_files = list(input_path.glob("*.png"))
        futures = [executor.submit(convert_png_to_jpeg2000, file, output_path) for file in input_files]

        # Display progress bar and wait for all tasks to complete
        for _ in tqdm(as_completed(futures), total=len(input_files)):
            pass

def main():
    if len(sys.argv) != 3:
        logging.warning("Usage: python png_to_jpeg2000.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    convert_to_jpeg2000(input_folder, output_folder)

if __name__ == "__main__":
    main()
