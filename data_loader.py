import tools.normalization_factors

import os
import time

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
def png_pipeline(data_dir=None, file_list=None, mode="training", downsample_factor=2, crop_w=256, crop_h=256,
                shard_id=0, num_shards=1):

    file_names, labels = fn.readers.file(
        file_root=data_dir,
        file_list=file_list,
        random_shuffle=(mode == 'training'),
        name="Reader",
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True)

    decoded_images = fn.decoders.image(
        file_names,
        device="mixed",
        output_type=types.RGB)

    #converted_images = fn.color_space_conversion(
    #    decoded_images,
    #    device="gpu",
    #    image_type=types.RGB,
    #    output_type=types.YCbCr)

    if mode == 'training':
        # Data Augmentations for Training:

        # Pick a random 256x256 crop from each 512x512 image
        # Mirror the images horizontally 40% of the time
        images = fn.crop_mirror_normalize(
            decoded_images,
            device="gpu",
            dtype=types.UINT8,
            crop=(crop_h, crop_w),
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            output_layout=types.NHWC,
            mirror=fn.random.coin_flip(probability=0.4))

        # 20% of the time, apply random brightness adjust betwen 50% and 120%
        brightness = 1.0 + fn.random.coin_flip(probability=0.2) * fn.random.uniform(range=(-0.5, 0.2))
        images = fn.brightness(images, device="gpu", brightness=brightness)

        # 50% of the time, apply a random rotation of 90, 180, or 270 degrees
        angle = fn.random.coin_flip(probability=0.5) * fn.random.uniform(range=(1, 4), dtype=dali.types.INT32) * 90.0
        images = fn.rotate(images, device="gpu", angle=angle)

        # Convert to NCHW
        normalized_full_images = fn.transpose(images, device="gpu", perm=[2, 0, 1])
    else:
        normalized_full_images = fn.crop_mirror_normalize(
            decoded_images,
            device="gpu",
            dtype=types.UINT8,
            crop=(crop_h, crop_w),
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            output_layout=types.NCHW,
            mirror=0)

    normalized_downsampled_images = fn.resize(
        normalized_full_images,
        device="gpu",
        interp_type=types.DALIInterpType.INTERP_CUBIC,
        size=[crop_w // downsample_factor, crop_h // downsample_factor])

    return labels, normalized_full_images, normalized_downsampled_images

class CustomDALIIterator(dali_torch.DALIGenericIterator):
    def __init__(self, pipelines, *args, **kwargs):
        super(CustomDALIIterator, self).__init__(pipelines, ["labels", "full", "down"], *args, **kwargs)

    def __next__(self):
        out = super().__next__()
        out = out[0]

        # Extract the downsampled and upsampled images from the output
        labels = out["labels"]
        full = out["full"]
        down = out["down"]

        return labels, full, down

class UpsamplingDataLoader:
    def __init__(self, batch_size, device_id, num_threads, seed, data_dir=None, file_list=None, mode='training',
                    downsample_factor=2, crop_w=256, crop_h=256, shard_id=1, num_shards=1):
        self.pipeline = png_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            data_dir=data_dir,
            file_list=file_list,
            mode=mode,
            downsample_factor=downsample_factor,
            crop_w=crop_w,
            crop_h=crop_h)
        self.pipeline.build()
        self.loader = CustomDALIIterator(
            [self.pipeline],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
