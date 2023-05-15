import argparse
import os
import re
import textwrap

def unique_subdir_names(path):
    # Get all subdirectories under the given path
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Compile a regex pattern to match numbers and trailing symbols at the end of the string
    pattern = re.compile(r'_clip_.*')

    # Collect unique names after removing numbers and trailing symbols
    unique_names = set()
    for subdir in subdirs:
        cleaned_name = pattern.sub('', subdir)
        unique_names.add(cleaned_name)

    return list(unique_names)

def main(args):
    unique_names = unique_subdir_names(args.path)

    unique_names.sort()

    for name in unique_names:
        print(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print unique subdirectory names without trailing numbers and symbols.")
    parser.add_argument("--path", type=str, default="~/dataset", help="Path to the folder containing subdirectories.")

    args = parser.parse_args()
    main(args)
