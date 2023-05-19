# Script to convert ONNX model to OpenVINO IR

# Same as the CLI command but easier to use:
# mo --input_model upsampling.onnx --input "input[1,3,1..,1..]{u8}" --output_dir openvino_model --compress_to_fp16

import os
import shutil

from openvino.tools.mo import convert_model

input_model = "upsampling.onnx"
input = "input[1,12,1..,1..]{u8}"
output_dir = "openvino_model"
model_name = "upsampling"
compress_to_fp16 = True

ov_model = convert_model(input_model=input_model, input=input, compress_to_fp16=True)



# Check if the output directory exists
if os.path.exists(output_dir):
    # If the directory exists, remove it using shutil.rmtree()
    shutil.rmtree(output_dir)



# This is taken from the openvino/tools/mo/main.py script so it behaves the same as the CLI tool:

import os
from openvino.tools.mo.pipeline.common import get_ir_version
from openvino.runtime import serialize

model_path_no_ext = os.path.normpath(os.path.join(output_dir, model_name))
model_path = model_path_no_ext + '.xml'

serialize(ov_model, model_path.encode('utf-8'), model_path.replace('.xml', '.bin').encode('utf-8'))

print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(model_path)))
print('[ SUCCESS ] XML file: {}'.format(model_path))
print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
