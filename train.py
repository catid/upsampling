import os
import shutil
import time
import random

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch._dynamo as dynamo

import deepspeed
from deepspeed import comm
from deepspeed import log_dist

import argparse

from upsampling_net import create_vapsr2x
from tiny_net import create_tiny2x

from data_loader import UpsamplingDataLoader
#from ffcv_data_loader import FfcvUpsamplingDataLoader

from deepspeed.runtime.config import DeepSpeedConfig

import nni

from torch.utils.tensorboard import SummaryWriter


def log_0(msg):
    log_dist(msg, ranks=[0])

def log_all(msg):
    log_dist(msg, ranks=[-1])


# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


def is_main_process():
    return comm.get_rank() == 0

def ref_forward_and_loss_fp16(criterion, data, target, model_engine):
    # DeepSpeed: forward + backward + optimize
    output = model_engine(data)

    # During training, the model produces NCHW FP16 -1..1 RGB images.
    # Target is a NCHW uint8 RGB image.  Convert to FP16 -1..1 RGB images.
    target = target.half()
    target = target / 127.5 - 1.0
    target = target.clamp(-1, 1)

    l1_loss = criterion.forward(output, target)
    return l1_loss

def ref_forward_and_loss_fp32(criterion, data, target, model_engine):
    # DeepSpeed: forward + backward + optimize
    output = model_engine(data)

    # During training, the model produces NCHW FP16 -1..1 RGB images.
    # Target is a NCHW uint8 RGB image.  Convert to FP16 -1..1 RGB images.
    target = target.float()
    target = target / 127.5 - 1.0
    target = target.clamp(-1, 1)

    l1_loss = criterion.forward(output, target)
    return l1_loss

def train_one_epoch(opt_forward_and_loss, criterion, train_loader, model_engine):
    train_loss = 0.0

    model_engine.train()

    with torch.set_grad_enabled(True):
        for batch_idx, (label, target, data) in enumerate(train_loader):
            data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)

            loss = opt_forward_and_loss(criterion, data, target, model_engine)

            model_engine.backward(loss)
            model_engine.step()

            train_loss += loss.item()

    return train_loss


def validation_one_epoch(opt_forward_and_loss, criterion, val_loader, model_engine):
    val_loss = 0.0

    model_engine.eval()

    with torch.set_grad_enabled(False):
        for batch_idx, (label, target, data) in enumerate(val_loader):
            data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)

            loss = opt_forward_and_loss(criterion, data, target, model_engine)

            val_loss += loss.item()

            if batch_idx == 0:
                test_images = data[:2]
                result_images = model_engine(test_images)
                example_images = (test_images, target[:2], result_images)

    return val_loss, example_images

def dict_compare(dict1, dict2):
    # Check if the dictionaries have the same length
    if len(dict1) != len(dict2):
        return False

    # Check if the dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Check if the dictionaries have the same values for each key
    for key in dict1:
        if dict1[key] != dict2[key]:
            return False

    return True

def delete_folder_contents(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def main(args):
    if args.nni:
        params = nni.get_next_parameter()
    else:
        params = {}
        params['learning_rate'] = 0.001

    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)

    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    # Model and optimizer
    model = create_tiny2x(d2sinput=False, rgb8output=False)

    torch._dynamo.config.verbose = False
    torch._dynamo.config.suppress_errors = True

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        #config_params=args.deepspeed_config,  <- This should be in the args
        model_parameters=model.parameters())

    log_dir = f"{args.log_dir}/upsampling"
    os.makedirs(log_dir, exist_ok=True)
    if args.reset:
        log_0("Resetting training - deleting Tensorboard directory")
        delete_folder_contents(log_dir)
    tensorboard = SummaryWriter(log_dir=log_dir)

    fp16 = model_engine.fp16_enabled()
    log_0(f'model_engine.fp16_enabled={fp16}')

    rank = model_engine.local_rank
    shard_id = model_engine.global_rank
    num_gpus = model_engine.world_size
    train_batch_size = model_engine.train_batch_size()
    data_loader_batch_size = model_engine.train_micro_batch_size_per_gpu()
    steps_per_print = model_engine.steps_per_print()

    log_all(f"rank = {rank}, num_shards = {num_gpus}, shard_id={shard_id}, train_batch_size = {train_batch_size}, data_loader_batch_size = {data_loader_batch_size}, steps_per_print = {steps_per_print}")

    num_loader_threads = os.cpu_count()//2
    seed = 0
    downsample_factor = 2
    crop_w = 256
    crop_h = 256

    if True:
        train_loader = UpsamplingDataLoader(
            batch_size=data_loader_batch_size,
            device_id=rank,
            num_threads=num_loader_threads,
            seed=seed,
            file_list=os.path.join(args.dataset_dir, "training_file_list.txt"),
            mode='training',
            downsample_factor=downsample_factor,
            crop_w=crop_w,
            crop_h=crop_h,
            shard_id=shard_id,
            num_shards=num_gpus)

        val_loader = UpsamplingDataLoader(
            batch_size=data_loader_batch_size,
            device_id=rank,
            num_threads=num_loader_threads,
            seed=seed,
            file_list=os.path.join(args.dataset_dir, "validation_file_list.txt"),
            mode='validation',
            downsample_factor=downsample_factor,
            crop_w=crop_w,
            crop_h=crop_h,
            shard_id=shard_id,
            num_shards=num_gpus)
    else:
        train_loader = FfcvUpsamplingDataLoader(
            batch_size=data_loader_batch_size,
            device_id=rank,
            num_threads=num_loader_threads,
            seed=seed,
            data_file=str(Path.home() / "ffcv_dataset"),
            mode='training',
            downsample_factor=downsample_factor,
            crop_w=crop_w,
            crop_h=crop_h,
            shard_id=shard_id,
            num_shards=num_gpus)

        val_loader = FfcvUpsamplingDataLoader(
            batch_size=data_loader_batch_size,
            device_id=rank,
            num_threads=num_loader_threads,
            seed=seed,
            data_file=str(Path.home() / "ffcv_dataset"),
            mode='validation',
            downsample_factor=downsample_factor,
            crop_w=crop_w,
            crop_h=crop_h,
            shard_id=shard_id,
            num_shards=num_gpus)

    # Loss functions

    if args.mse:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    criterion.cuda(rank)

    if fp16:
        forward_and_loss = ref_forward_and_loss_fp16
    else:
        forward_and_loss = ref_forward_and_loss_fp32
    forward_and_loss = dynamo.optimize("inductor")(forward_and_loss)

    # Initialize training

    best_val_loss = float("inf")
    avg_val_loss = float("inf")
    start_epoch = 0
    epochs_without_improvement = 0

    if args.reset:
        log_0("Resetting training - deleting output directory")
        if rank == 0:
            delete_folder_contents(args.output_dir)
    else:
        _, client_state = model_engine.load_checkpoint(load_dir=args.output_dir)
        if client_state is not None:
            start_epoch = client_state['epoch'] + 1
            if client_state['crop_w'] != crop_w or client_state['crop_h'] != crop_h or client_state['train_version'] != 1:
                log_all(f"Model checkpoint is incompatible with current training parameters. Please reset the training by deleting the output directory {args.output_dir} or running with --reset")
                exit(1)
            if not dict_compare(params, client_state['model_params']):
                log_all(f"Model params is incompatible with current training parameters. Please reset the training by deleting the output directory {args.output_dir} or running with --reset")
                exit(1)
            avg_val_loss = client_state['avg_val_loss']
            best_val_loss = avg_val_loss
            log_all(f"Loaded checkpoint at epoch {client_state['epoch']}")
        else:
            log_all("No checkpoint found - Starting training from scratch")

    # Training/validation loop

    for epoch in range(start_epoch, args.max_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(forward_and_loss, criterion, train_loader, model_engine)

        val_loss, example_images = validation_one_epoch(forward_and_loss, criterion, val_loader, model_engine)

        end_time = time.time()
        epoch_time = end_time - start_time

        if args.nni:
            nni.report_intermediate_result(avg_val_loss)

        # Sync variables between machines
        sum_train_loss = torch.tensor(train_loss).cuda(rank)
        sum_val_loss = torch.tensor(val_loss).cuda(rank)
        comm.all_reduce(tensor=sum_train_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_val_loss, op=comm.ReduceOp.SUM)

        total_train_items = len(train_loader) * num_gpus
        total_val_items = len(val_loader) * num_gpus
        comm.barrier()
        avg_train_loss = sum_train_loss.item() / total_train_items
        avg_val_loss = sum_val_loss.item() / total_val_items

        if is_main_process():
            events = [
                ("AvgTrainLoss", avg_train_loss, model_engine.global_samples),
                ("AvgValLoss", avg_val_loss, model_engine.global_samples)
            ]
            for event in events:
                tensorboard.add_scalar(*event)

            # Note: Tensorboard needs NCHW uint8
            input_images = example_images[0]
            target_images = example_images[1]

            # During training, the model produces FP16 -1..1 RGB images.
            # Convert to NCHW uint8 0..255 RGB images for display.
            output_images = example_images[2]

            output_images = output_images * 127.5 + 128
            output_images = torch.clamp(output_images, 0, 255)
            output_images = output_images.to(torch.uint8)

            tensorboard.add_images('input', input_images, epoch)
            tensorboard.add_images('target', target_images, epoch)
            tensorboard.add_images('output', output_images, epoch)

            log_0(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

        # Check if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            log_0(f'New best validation loss: {best_val_loss:.4f}')

            client_state = {
                'train_version': 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'epoch': epoch,
                'crop_w': crop_w,
                'crop_h': crop_h,
                'fp16': fp16,
                'model_params': params
            }
            model_engine.save_checkpoint(save_dir=args.output_dir, client_state=client_state)
            log_all(f'Saved new best checkpoint')
        else:
            epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= args.patience:
                log_0(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss}")
                break

    if is_main_process():
        log_0(f'Training complete.  Final validation loss: {avg_val_loss}')

    # Report final validation loss
    if args.nni:
        nni.report_final_result(avg_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset-dir", type=str, default=str(Path.home() / "dataset"), help="Path to the dataset directory (default: ~/dataset/)")
    parser.add_argument("--max-epochs", type=int, default=1000000, help="Maximum epochs to train")
    parser.add_argument("--patience", type=int, default=200, help="Patience for validation loss not decreasing before early stopping")
    parser.add_argument("--nni", action="store_true", help="Enable NNI mode")
    parser.add_argument("--output-dir", type=str, default="output_model", help="Path to the output trained model")
    parser.add_argument("--log-dir", type=str, default="tb_logs", help="Path to the Tensorboard logs")
    parser.add_argument("--reset", action="store_true", help="Reset training from scratch")
    parser.add_argument("--mse", action="store_true", help="Use MSE instead of L1 loss")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.deepspeed_config==None or len(args.deepspeed_config)==0:
        args.deepspeed_config = "deepspeed_config.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
