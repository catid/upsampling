import sys
import os

# Add the parent folder to the Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import subprocess
from pathlib import Path

from tools.logging_tools import setup_colored_logging
import logging
setup_colored_logging()

def change_extension_to_mp4(output_dir, file_name):
    # Split the file_name into its name and extension
    file_base, file_ext = os.path.splitext(file_name)
    
    # Replace the extension with .mp4
    new_file_name = file_base + ".mp4"
    
    # Join the output directory and the new file name
    output_file_path = os.path.join(output_dir, new_file_name)
    
    return output_file_path

def convert_mkv_to_mp4(args):
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.mkv'):
                input_file_path = Path(root) / file
                output_file_path = change_extension_to_mp4(args.output_dir, file)

                if os.path.isfile(output_file_path):
                    logging.warning(f"Skipping remuxing file that already exists: {output_file_path}")
                    continue

                logging.info(f"Working on: {input_file_path} -> {output_file_path}")

                command = [
                    'ffmpeg',
                    '-y',
                    '-i', str(input_file_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '256k',
                    '-movflags', '+faststart',
                    str(output_file_path)
                ]
                subprocess.run(command, check=True)

                logging.info(f"Success! {input_file_path} -> {output_file_path}")

def main(args):
    convert_mkv_to_mp4(args)

    logging.info(f"Successfully converted all movies")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .mkv files to .mp4 without re-encoding.')
    parser.add_argument('input_dir', type=str, help='Path to the movies folder containing .mkv files.')
    parser.add_argument('output_dir', type=str, help='Output path where .mp4 files will be written')

    args = parser.parse_args()
    main(args)
