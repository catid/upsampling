import os
import argparse

import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map
from pathlib import Path
from functools import partial
import multiprocessing

from tools.logging_tools import setup_colored_logging
import logging
setup_colored_logging()


def image_stats(args, image_path):
    img = Image.open(image_path)
    img_array = np.asarray(img, dtype=np.float64)

    # Separate RGB channels
    R_channel = img_array[:, :, 0]
    G_channel = img_array[:, :, 1]
    B_channel = img_array[:, :, 2]

    # Function to compute mean and std_dev for each channel
    def channel_stats(channel):
        # Compensated summation for mean
        mean = 0
        delta = 0
        for value in np.nditer(channel):
            y = value - delta
            t = mean + y
            delta = (t - mean) - y
            mean = t

        mean /= channel.size

        # Compensated summation for variance
        variance = 0
        delta = 0
        for value in np.nditer(channel):
            y = (value - mean) ** 2 - delta
            t = variance + y
            delta = (t - variance) - y
            variance = t

        variance /= channel.size
        std_dev = np.sqrt(variance)

        return mean, std_dev

    return {
        'R': channel_stats(R_channel),
        'G': channel_stats(G_channel),
        'B': channel_stats(B_channel),
    }


def overall_stats(results):
    channel_sum = {'R': 0.0, 'G': 0.0, 'B': 0.0}
    channel_sum_squares = {'R': 0.0, 'G': 0.0, 'B': 0.0}

    for stats in results:
        for channel in ['R', 'G', 'B']:
            mean, std_dev = stats[channel]
            channel_sum[channel] += mean
            channel_sum_squares[channel] += (std_dev ** 2 + mean ** 2)

    overall_mean = {channel: channel_sum[channel] / len(results) for channel in ['R', 'G', 'B']}
    overall_std_dev = {channel: np.sqrt(channel_sum_squares[channel] / len(results) - overall_mean[channel] ** 2) for channel in ['R', 'G', 'B']}

    return overall_mean, overall_std_dev


def main():
    parser = argparse.ArgumentParser(description="Measure mean/std_dev for RGB channels in the dataset")
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser("~/dataset"), help='Data directory to search for .png images')

    args = parser.parse_args()

    all_png_files = list(Path(args.data_dir).rglob("*.png"))

    logging.info(f"Processing {len(all_png_files)} .PNG images...")

    # Partially apply the additional arguments to process_image
    image_stats_with_args = partial(image_stats, args)

    # Use process_map to handle the progress bar and processing
    results = process_map(image_stats_with_args, all_png_files, chunksize=2, desc="Calculating")

    mean, std_dev = overall_stats(results)

    mean = {channel: mean[channel] / 255 for channel in ['R', 'G', 'B']}
    std = {channel: std_dev[channel] / 255 for channel in ['R', 'G', 'B']}

    logging.info(f"mean = [{mean['R']} * 255, {mean['G']} * 255, {mean['B']} * 255]")
    logging.info(f"std = [{std['R']} * 255, {std['G']} * 255, {std['B']} * 255]")

if __name__ == '__main__':
    main()
