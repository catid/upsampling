import argparse
import os
import time

from pathlib import Path
import numpy as np
from PIL import Image

from openvino.preprocess import PrePostProcessor, ColorFormat
import openvino.runtime as ov

def process_image(compiled_model, input_path, output_path, w, h):
    # Load an image with PIL
    input_image = Image.open(input_path)
    # Convert to numpy array
    input_image_np = np.array(input_image)

    input_image_np = input_image_np[:h, :w]

    # Reshape input
    input_image_np = input_image_np.transpose((2, 0, 1))
    input_image_np = np.expand_dims(input_image_np, axis=0)

    # The output would be your upscaled image
    #print("Inference input shape:", input_image_np.shape)

    t0 = time.time()

    infer_request = compiled_model.create_infer_request()
    infer_request.infer([input_image_np])
    output = infer_request.get_output_tensor().data

    t1 = time.time()
    print(f"Inference time: {(t1 - t0)*1000.0:.3f} milliseconds")

    # Remove batch dimension
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
    w = 256
    h = 256

    model.reshape({'input': [1, 3, h, w]})

    print(f"Model input names: {model.input().names}, shape: {model.input().partial_shape}")
    print(f"Model output names: {model.output().names}, shape: {model.output().partial_shape}")

    ppp = PrePostProcessor(model)

    # FIXME: Can we change layout here?

    ppp.input("input").tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout('NCHW')) \
            .set_color_format(ColorFormat.RGB)
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
