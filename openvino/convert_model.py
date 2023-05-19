# Script to convert ONNX model to OpenVINO IR

# Same as the CLI command but easier to use:
# mo --input_model upsampling.onnx --input "input[1,3,1..,1..]{u8}" --output_dir openvino_model --compress_to_fp16

from openvino.tools.mo import convert_model

input_model = "upsampling.onnx"
input = "input[1,3,1..,1..]{u8}"
output_dir = "openvino_model"
compress_to_fp16 = True

#output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
#model_path_no_ext = os.path.normpath(os.path.join(output_dir, argv.model_name))
#model_path = model_path_no_ext + '.xml'

ov_model = convert_model(input_model=input_model, input=input, output=output_dir, compress_to_fp16=True)
