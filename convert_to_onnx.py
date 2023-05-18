import torch
import argparse
import onnx
from onnxconverter_common import float16

from upsampling_net import create_vapsr2x
from joint_net import create_joint2x

import logging
from tools.logging_tools import setup_colored_logging
setup_colored_logging()

def load_model(model_path):
    model = create_joint2x(rgb8output=True)
    model.load_state_dict(torch.load(model_path))
    model.half().eval().cuda()
    return model

def main(args):
    logging.info("Loading model...")

    dummy_input = torch.randn(1, 3, 224, 224).byte().cuda()

    model = load_model(args.input)

    logging.info("Converting model to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        args.temp,
        verbose=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},    # variable length axes
                    'output' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}})

    logging.info("Converting model to FP16 ONNX...")

    model_fp16 = float16.convert_float_to_float16_model_path(args.temp)

    onnx.save(model_fp16, args.output)

    logging.info("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--input", type=str, default="upsampling.pth", help="Input model")
    parser.add_argument("--temp", type=str, default="temp.onnx", help="Temp intermediate model file")
    parser.add_argument("--output", type=str, default="upsampling.onnx", help="Output model")

    args = parser.parse_args()

    main(args)
