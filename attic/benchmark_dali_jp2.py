import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

import cupy as cp


# Define the custom PyTorch operator
def rgb_to_oklab(x):
    # Input is 0..255 RGB
    m1 = cp.array([
        [0.4122214708 / 255.0, 0.5363325363 / 255.0, 0.0514459929 / 255.0],
        [0.2119034982 / 255.0, 0.6806995451 / 255.0, 0.1073969566 / 255.0],
        [0.0883024619 / 255.0, 0.2817188376 / 255.0, 0.6299787005 / 255.0]
    ], device=x.device)

    m2 = cp.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ], device=x.device)

    lms = cp.matmul(x, m1.T)
    lms_cbrt = cp.cbrt(lms) # Cube root
    lab = cp.matmul(lms_cbrt, m2.T)

    return lab


import os
import time

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy



@pipeline_def
def jp2_pipeline(data_dir, mode="training", downsample_factor=0.25,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255]):

    file_names, labels = fn.readers.file(file_root=data_dir, file_list=None, random_shuffle=(mode == 'training'), name="Reader")
    shapes = fn.peek_image_shape(file_names)
    size = fn.slice(shapes, 0, 2, axes=[0])  # remove the channel axis

    decoded_images = fn.decoders.image(
        file_names,
        device="mixed",
        output_type=types.RGB)
    normalized_full_images = fn.crop_mirror_normalize(
        decoded_images, device="gpu",
        dtype=types.FLOAT,
        crop=None,
        output_layout=types.NCHW,
        mean=mean,
        std=std,
        mirror=(mode == 'training'))
    normalized_downsampled_images = fn.resize(
        normalized_full_images,
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
        self.pipeline = jp2_pipeline(
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
    data_dir = '/home/catid/upsampling1/data'
    batch_size = 64
    downsample_factor = 0.25 # Should be 1/2 or 1/4 or some other power of two

    for epoch in range(10):  # Example number of epochs
        train_loader = DALIDataLoader(batch_size=batch_size, device_id=0, num_threads=8, seed=epoch, data_dir=data_dir, mode='training', downsample_factor=downsample_factor)
        val_loader = DALIDataLoader(batch_size=batch_size, device_id=0, num_threads=8, seed=epoch, data_dir=data_dir, mode='validation', downsample_factor=downsample_factor)

        for i, (labels, f16_full, f16_down) in enumerate(train_loader):
            for j in range(f16_full.size(0)):
                full = f16_full[j]
                down = f16_down[j]
                logging.info(f"Training: Pair {down.shape} -> {full.shape} ({labels[j]})")

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
