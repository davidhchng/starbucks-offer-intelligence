import pickle
import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# zipmap=False forces probability output to be a plain float tensor
options = {id(model): {'zipmap': False}}
initial_type = [('float_input', FloatTensorType([None, 12]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)

onnx.save(onnx_model, 'model.onnx')
print("Outputs:", [o.name for o in onnx_model.graph.output])
