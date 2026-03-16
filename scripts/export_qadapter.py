import torch
import onnxruntime as ort
import numpy as np

# Adjust import path to match your RRA project structure
from rra.models import QFormer

print("[AegisRad] Loading RRA-Q Adapter...")
qadapter = QFormer.from_pretrained("path/to/rra/checkpoint")
qadapter.eval()

dummy = torch.randn(1, 2048, 7, 7)

print("[AegisRad] Exporting to ONNX...")
torch.onnx.export(
    qadapter,
    dummy,
    "models/rra_qadapter.onnx",
    input_names=["visual_features"],
    output_names=["query_embeddings"],
    dynamic_axes={"visual_features": {0: "batch"}},
    opset_version=14
)

print("[AegisRad] Validating export...")
sess = ort.InferenceSession("models/rra_qadapter.onnx")
out  = sess.run(None, {"visual_features": dummy.numpy()})
print(f"Output shape: {out[0].shape}")
# Expected: (1, 32, 768)
print("[AegisRad] RRA-Q Adapter export complete.")