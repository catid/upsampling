import os
import time
import sys

# Add the parent folder to the Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools.normalization_factors

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import nvidia.dali.backend as backend

import multiprocessing

import cupy as cp

import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()


@pipeline_def(batch_size=64, num_threads = 8, exec_async=False, exec_pipelined=False)
def png_pipeline(data_dir, mode="training", downsample_factor=0.25,
                mean=tools.normalization_factors.mean,
                std=tools.normalization_factors.std):

    file_names, labels = fn.readers.file(file_root=data_dir, file_list=None, random_shuffle=(mode == 'training'), name="Reader")
    shapes = fn.peek_image_shape(file_names)
    size = fn.slice(shapes, 0, 2, axes=[0])  # remove the channel axis

    decoded_images = fn.decoders.image(
        file_names,
        device="mixed",
        output_type=types.RGB)

    cropped = fn.crop(
        decoded_images,
        crop_h=256,
        crop_w=256,
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
        dtype=types.FLOAT,
        out_of_bounds_policy="error",
        rounding="truncate")

    normalized_full_images = fn.crop_mirror_normalize(
        cropped,
        device="gpu",
        dtype=types.FLOAT,
        crop=None,
        output_layout=types.NCHW,
        mean=mean,
        std=std,
        mirror=(mode == 'training'))

    normalized_downsampled_images = fn.resize(
        normalized_full_images,
        device="gpu",
        interp_type=types.DALIInterpType.INTERP_CUBIC,
        size=size * downsample_factor)

    f16_full = fn.cast(normalized_full_images, dtype=types.DALIDataType.FLOAT16)
    f16_down = fn.cast(normalized_downsampled_images, dtype=types.DALIDataType.FLOAT16)

    return labels, f16_full, f16_down

class CustomDALIIterator(dali_torch.DALIGenericIterator):
    def __init__(self, pipelines, *args, **kwargs):
        super(CustomDALIIterator, self).__init__(pipelines, ["labels", "f16_full", "f16_down"], *args, **kwargs)

    def __next__(self):
        out = super().__next__()
        out = out[0]

        # Extract the downsampled and upsampled images from the output
        labels = out["labels"]
        f16_full = out["f16_full"]
        f16_down = out["f16_down"]

        return labels, f16_full, f16_down

class DALIDataLoader:
    def __init__(self, batch_size, device_id, num_threads, seed, data_dir, mode='training', downsample_factor=0.25):
        self.pipeline = png_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            data_dir=data_dir,
            mode=mode,
            downsample_factor=downsample_factor)
        self.pipeline.build()
        self.loader = CustomDALIIterator(
            [self.pipeline],
            reader_name='Reader',
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)


def main():
    data_dir = os.path.expanduser("~/dataset")
    batch_size = 64
    downsample_factor = 0.25 # Should be 1/2 or 1/4 or some other power of two

    num_threads = multiprocessing.cpu_count()
    device_id = 0

    for epoch in range(10):  # Example number of epochs
        train_loader = DALIDataLoader(batch_size=batch_size, device_id=device_id, num_threads=num_threads, seed=epoch, data_dir=data_dir, mode='training', downsample_factor=downsample_factor)
        val_loader = DALIDataLoader(batch_size=batch_size, device_id=device_id, num_threads=num_threads, seed=epoch, data_dir=data_dir, mode='validation', downsample_factor=downsample_factor)

        for i, (labels, f16_full, f16_down) in enumerate(train_loader):
            for j in range(f16_full.size(0)):
                full = f16_full[j]
                down = f16_down[j]
                #logging.info(f"Training: Pair {down.shape} -> {full.shape} ({labels[j]})")

            if i <= 1:
                start_time = time.time()
            else:
                end_time = time.time()
                logging.info(f"{((i + 1 - 1) * batch_size) / (end_time - start_time)}")

        for i, (labels, f16_full, f16_down) in enumerate(val_loader):
            for j in range(f16_full.size(0)):
                full = f16_full[j]
                down = f16_down[j]
                #logging.info(f"Validation: Pair {down.shape} -> {full.shape}")

if __name__ == '__main__':
    main()
