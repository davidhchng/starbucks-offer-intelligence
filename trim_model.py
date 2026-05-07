from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
import onnx

model = onnx.load('model.onnx')
trimmed = select_model_inputs_outputs(model, outputs=['output_label'])
onnx.save(trimmed, 'model.onnx')
print("Done. Outputs:", [o.name for o in trimmed.graph.output])
