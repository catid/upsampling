# This script extracts data from 2011 imagenet torrent

import tarfile
import io
import os
import argparse
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from threading import Semaphore
from tqdm import tqdm

from tools.logging_tools import setup_colored_logging
import logging
setup_colored_logging()

def extract_internal_tarfile(args, semaphore, name, internal_tar_bytes):
    #logging.info(f"{name}: Opening tar from memory...")

    # Open the internal .tar file from memory
    with tarfile.open(fileobj=internal_tar_bytes) as internal_tar:
        target_dir = os.path.join(args.output, os.path.splitext(name)[0])
        os.makedirs(target_dir, exist_ok=True)

        #logging.info(f"{name}: Writing {target_dir}")

        # Extract the contents of the internal .tar file into the target folder
        internal_tar.extractall(target_dir)

    semaphore.release()

def main(args):
    # Open the outer .tar file
    with tarfile.open(args.input_tar_path) as outer_tar:
        # Filter out the .tar files within the outer .tar file
        internal_tar_infos = [info for info in outer_tar.getmembers() if info.name.endswith('.tar')]

        semaphore = Semaphore(args.num_threads)

        logging.info(f"Starting to read .tar files contained within {args.input_tar_path}")

        num_tasks = len(internal_tar_infos)

        # Create a ThreadPoolExecutor to handle the tasks
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Use tqdm to display progress
            with tqdm(total=num_tasks) as progress_bar:

                futures = []

                for tar_info in internal_tar_infos:

                    semaphore.acquire()

                    # Read the internal .tar file into memory
                    internal_tar_bytes = io.BytesIO(outer_tar.extractfile(tar_info).read())

                    future = executor.submit(extract_internal_tarfile, args, semaphore, tar_info.name, internal_tar_bytes)
                    futures.append(future)

                    future.add_done_callback(lambda x: progress_bar.update())

                results = [future.result() for future in futures]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a .tar file and extract its contents.')
    parser.add_argument('input_tar_path', default='fall11_whole.tar', help='Path to the input .tar file.')
    parser.add_argument('--output', default=os.path.expanduser("~/dataset"), help='Path to the output folder.')
    parser.add_argument('--num_threads', default=multiprocessing.cpu_count(), help='Number of threads used for workers.')

    args = parser.parse_args()

    main(args)
