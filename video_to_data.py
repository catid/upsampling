# This script reads a directory of video files (produced as described in the README)
# and decodes them in parallel, sampling frames from the videos at regular intervals,
# and writing them to disk as PNG images for training.

import os
import av
from threading import Semaphore
import multiprocessing
import concurrent
from concurrent.futures import ThreadPoolExecutor

from tqdm.contrib.concurrent import process_map
from functools import partial

from tools.dataset_tools import save_random_image_crops_to_disk, create_super_random_threadsafe_generator

import argparse
import shutil
import subprocess

from tools.logging_tools import setup_colored_logging
import logging
setup_colored_logging()

logging.getLogger('libav').setLevel(logging.ERROR)


class AVContainerContext:
    def __init__(self, file_path):
        self.file_path = file_path
        self.container = None
        self.video_stream = None

    def __enter__(self):
        hwaccel = 'cuvid'
        threads = 1

        options = {
            'hwaccel': hwaccel,
            'threads': str(threads)
        }

        self.container = av.open(self.file_path, options=options)

        # Find the first video stream in the container.
        for stream in self.container.streams:
            if stream.type == 'video':
                self.video_stream = stream
                break

        if self.video_stream is None:
            raise ValueError("No video stream found in the file")

        return self.container, self.video_stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self.container is not None:
            self.container.close()


def process_frame(semaphore, frame, label_output_dir, args):
    try:
        # Convert PyAV frame to PIL image
        img = frame.to_image()

        save_random_image_crops_to_disk(
            img,
            f"{frame.time:.2f}",
            label_output_dir,
            args.width,
            args.height,
            downsample_first=True)
    except Exception as e:
        logging.warning(f"Failed to process frame for {label_output_dir}: {e}")
    semaphore.release()

def process_video_file(args, input_file):
    #logging.info(f"Opening/demuxing video {input_file}...")

    rng = create_super_random_threadsafe_generator()

    semaphore = Semaphore(args.num_frame_threads)
    futures = []

    label = os.path.splitext(os.path.basename(input_file))[0]
    label_output_dir = os.path.join(args.output_dir, label)
    os.makedirs(label_output_dir, exist_ok=True)

    with AVContainerContext(input_file) as (container, video_stream):
        with ThreadPoolExecutor(max_workers=args.num_frame_threads) as executor:

            frame_index = 0
            previous_frame_index = -args.min_interval

            for packet in container.demux(video_stream):
                frames = packet.decode()

                # Iterate over each frame
                for frame in frames:
                    # Check if the current frame is at least args.min_separation apart from the previous selected frame
                    if frame_index - previous_frame_index >= args.min_interval:
                        # Randomly select frames based on args.frame_interval
                        if frame_index - previous_frame_index >= args.max_interval or rng.random() < 1 / args.frame_interval:
                            previous_frame_index = frame_index

                            # Dispatch processing for this frame
                            semaphore.acquire()
                            future = executor.submit(
                                process_frame, semaphore, frame, label_output_dir, args)
                            futures.append(future)

                    frame_index += 1

    #logging.info(f"Demuxed video {input_file}.  Processing frames...")

    concurrent.futures.wait(futures)

    #logging.info(f"Completed video {input_file}.")

def get_video_files(directory):
    video_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.mp4') or file_name.endswith('.avi') or file_name.endswith('.mkv'):
            video_files.append(os.path.join(directory, file_name))
    return video_files

def process_multiple_videos(args):
    files = get_video_files(args.split_dir)

    # Partially apply the additional arguments to process_image
    process_video_file_with_args = partial(process_video_file, args)

    # Use process_map to handle the progress bar and processing
    process_map(process_video_file_with_args, files, max_workers=args.num_video_threads, chunksize=1, desc="Videos")


def split_video(args):
    shutil.rmtree(args.split_dir, ignore_errors=True)

    label = os.path.splitext(os.path.basename(args.input_video))[0]

    # Create the output directory if it doesn't exist
    os.makedirs(args.split_dir, exist_ok=True)

    if any(os.listdir(args.split_dir)):
        logging.error(f"Files already exist under {args.split_dir} - Delete those before splitting another video")
        return

    logging.info(f"Launching ffmpeg to split {args.input_video} into {args.segment_time} second clips under {args.split_dir}")

    # Prepare the ffmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-fflags", "+igndts",
        "-i", args.input_video,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(args.segment_time),
        "-map", "0",
        os.path.join(args.split_dir, f"{label}_clip_%03d.mkv")
    ]

    # Run the ffmpeg command
    subprocess.run(ffmpeg_command, check=True)

    if not os.path.exists(args.split_dir):
        raise Exception("You have to split the file into segments first. Check the README")
    if not any(os.listdir(args.split_dir)):
        raise Exception("You have to split the file into segments first. Check the README")

    logging.info("Video split complete.")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video segments")
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument("--segment_time", type=int, default=120, help="Duration of each segment in seconds (Default: 120 seconds).")
    parser.add_argument("--split-dir", default="output_split_video", help="Intermediate folder that will be deleted (default: output_split_video)")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/dataset"), help="Output frames directory (default: data)")
    parser.add_argument("--frame-interval", type=int, default=10, help="Average frame interval (default: 10).  Randomly selects frames at about this rate")
    parser.add_argument("--min-interval", type=int, default=4, help="Min frame interval (default: 4).  Minimum interval between selected random frames")
    parser.add_argument("--max-interval", type=int, default=15, help="Max frame interval (default: 15).  Maximum interval between selected random frames")
    parser.add_argument("--num-video-threads", type=int, default=multiprocessing.cpu_count() * 2 // 3, help="Number of videos to process in parallel (default: CPU count // 2)")
    parser.add_argument("--num-frame-threads", type=int, default=2, help="Number of frames to process in parallel (default: 2)")
    parser.add_argument("--width", type=int, default=512, help="Crop width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Crop height (default: 512)")

    args = parser.parse_args()

    logging.info(f"Using output dir: {args.output_dir}")

    split_video(args)

    os.makedirs(args.output_dir, exist_ok=True)

    process_multiple_videos(args)

    logging.info(f"Video successfully incorporated into the dataset")

if __name__ == "__main__":
    main()
