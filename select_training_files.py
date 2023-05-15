import os
import random
import argparse
from pathlib import Path

import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

def create_file_lists(args):
    logging.info(f"Enumerating dataset files under {args.dataset_dir}...")

    all_files = [os.path.join(root, f) for root, dirs, files in os.walk(args.dataset_dir) for f in files if f.lower().endswith(('.jpeg', '.png', '.jpg'))]

    logging.info(f"Found {len(all_files)} total image files")

    num_train = args.max_training_files
    num_val = args.max_validation_files

    if num_train + num_val > len(all_files):
        ratio = len(all_files) / (num_train + num_val)
        num_train = (num_train * len(all_files)) // (num_train + num_val)
        num_val = len(all_files) - num_train

    logging.info(f"Selecting {num_train} files for training ({num_train * 100.0 / (num_train + num_val)}%).")
    logging.info(f"Selecting {num_val} files for validation ({num_val * 100.0 / (num_train + num_val)}%).")

    logging.info("Shuffling...")

    random.shuffle(all_files)

    logging.info("Writing training file list...")

    training_files = all_files[:num_train]

    logging.info(f"Selected {len(training_files)} training files")

    with open(os.path.join(args.dataset_dir, "training_file_list.txt"), "w") as f:
        for index, file in enumerate(training_files):
            f.write(f"{os.path.join(args.dataset_dir, file)} {index}\n")

    logging.info("Writing validation file list...")

    validation_files = all_files[num_train:num_train + num_val]

    logging.info(f"Selected {len(validation_files)} validation files")

    with open(os.path.join(args.dataset_dir, "validation_file_list.txt"), "w") as f:
        for index, file in enumerate(validation_files):
            f.write(f"{os.path.join(args.dataset_dir, file)} {index}\n")

    logging.info("Operation completed successfully.  Training can now be started")

def main():
    parser = argparse.ArgumentParser(description="Generate file lists for DALI fn.readers.file")
    parser.add_argument("--dataset-dir", type=str, default=str(Path.home() / "dataset"),
                        help="Path to the dataset directory (default: ~/dataset/)")
    parser.add_argument("--max-training-files", type=int, default=1000000,
                        help="Maximum number of files to include in the training set")
    parser.add_argument("--max-validation-files", type=int, default=10000,
                        help="Maximum number of files to include in the validation set")

    args = parser.parse_args()

    create_file_lists(args)

if __name__ == "__main__":
    main()
