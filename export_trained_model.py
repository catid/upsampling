import torch
import deepspeed

import argparse

from upsampling_net import create_vapsr2x
from tiny_net import create_tiny2x

import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

def main(args):
    # Initialize your model
    model = create_tiny2x(d2sinput=True, rgb8output=True)

    fp16 = not args.fp32

    logging.info(f"Loading model from {args.model_dir} fp16={fp16}")

    config_dict = {
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": fp16,
            "initial_scale_power": 8
        },
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 1,
    }

    ds_model, _, _, _ = deepspeed.initialize(config=config_dict,
                                             model=model,
                                             model_parameters=model.parameters(),
                                             optimizer=None)

    # Load Deepspeed checkpoint
    ds_model.load_checkpoint(load_dir=args.model_dir)

    saved_state_dict = ds_model.state_dict()

    fixed_state_dict = {key.replace("module.", ""): value for key, value in saved_state_dict.items()}

    torch.save(fixed_state_dict, args.output)

    logging.info(f"Wrote model to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model-dir", type=str, default="output_model", help="Path to the trained model")
    parser.add_argument("--output", type=str, default="upsampling.pth", help="Path to the output PyTorch .pth model file")
    parser.add_argument("--fp32", action="store_true", help="Expect FP32 model")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    main(args)
