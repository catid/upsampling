import argparse
import os
import time

import einops
from pathlib import Path
import numpy as np
from PIL import Image

from openvino.preprocess import PrePostProcessor, ColorFormat
import openvino.runtime as ov

def process_image(compiled_model, input_path, output_path, w, h):
    # Load an image with PIL
    input_image = Image.open(input_path)
    input_image = input_image.resize((1920//2, 1080//2))
    # Convert to numpy array
    input_image_np = np.array(input_image)

    # Reshape input from RGB with shape HxWxC to model input shape Bx(C*4)xHxW.
    # Note this is also doing a PixelUnshuffle(2) operation on the image, which is more efficient than doing it inside the model.
    # However, normalization is more efficient to do inside the model.
    input_image_np = np.expand_dims(input_image_np, axis=0)
    input_image_np = einops.rearrange(input_image_np, 'b (h p1) (w p2) c -> b (c p1 p2) h w', p1=2, p2=2)

    # The output would be your upscaled image
    print("Inference input shape:", input_image_np.shape)

    t0 = time.time()

    infer_request = compiled_model.create_infer_request()
    infer_request.infer([input_image_np])
    output = infer_request.get_output_tensor().data

    t1 = time.time()
    print(f"Inference time: {(t1 - t0)*1000.0:.3f} milliseconds")

    # Reshape 8-bit RGB model output shape BxCxHxW to standard RGB image shape HxWxC
    output = np.squeeze(output, axis=0)
    output = output.transpose((1, 2, 0))

    # The output would be your upscaled image
    #print("Inference output shape:", output.shape)

    # Save output to image
    output_img_pil = Image.fromarray(output)
    output_img_pil.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='2x Image Upsampling with OpenVINO')
    parser.add_argument('--model-dir', type=str, default="openvino_model", help='Path to OpenVino model directory')
    parser.add_argument('--input-dir', type=str, default="urban100", help='Path to input image directory')
    parser.add_argument('--output-dir', type=str, default="urban200", help='Path to output image directory')
    parser.add_argument('--device', type=str, default="GPU", help='OpenVINO device to use')
    args = parser.parse_args()

    # Run inference
    model_xml = os.path.join(args.model_dir, "upsampling.xml")
    model_bin = os.path.join(args.model_dir, "upsampling.bin")

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Create cache folder
    cache_folder = Path("model_cache")
    cache_folder.mkdir(exist_ok=True)

    # Initialize OpenVINO runtime
    core = ov.Core()
    core.set_property({'CACHE_DIR': cache_folder})

    # print available devices
    for device in core.available_devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"Detected OpenVINO device: {device} ({device_name})")

    print("Loading model...")

    # Read the model
    model = core.read_model(model_xml, model_bin)

    # Note: Intel GPU only supports fixed-size input
    # https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU.html#dynamic-shapes
    w = 1920//2
    h = 1080//2

    # Note the model input is expected to be in a weird shape.
    # First off, it should be NCHW, not NHWC.  This is opposite the usual layout of RGB images.
    # Secondly, the model expects that the CPU has effectively already performed:
    # F.pixel_unshuffle(rgb, downscale_factor=2)
    # This is reasonable because moving memory arround to get to BCHW has to be done anyway,
    # so we might as well do the PixelUnshuffle operation at the same time.
    # This means it does not need to be done in the critical path on GPU, where it's actually a
    # fairly expensive operation.
    model.reshape({'input': [1, 12, h//2, w//2]})

    print(f"Model input names: {model.input().names}, shape: {model.input().partial_shape}")
    print(f"Model output names: {model.output().names}, shape: {model.output().partial_shape}")

    ppp = PrePostProcessor(model)

    # FIXME: Can we change layout here?

    ppp.input("input").tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout('NCHW'))
    # Note: .set_memory_type("GPU_SURFACE") seems to break the model somehow.

    ppp.input("input").model().set_layout(ov.Layout('NCHW'))

    ppp.output('output').tensor() \
        .set_layout(ov.Layout("NCHW"))

    model = ppp.build()

    #compiled_model = core.compile_model(model, args.device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    compiled_model = core.compile_model(model, args.device, {"PERFORMANCE_HINT": "LATENCY"})

    print("Running model...")

    # Process each image in the input directory
    start_time = time.time()
    files = sorted(os.listdir(args.input_dir))
    for filename in files:
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            process_image(compiled_model, input_path, output_path, w, h)
            print(f"Wrote: {output_path}")
    end_time = time.time()

    print(f"Processed all images in {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
