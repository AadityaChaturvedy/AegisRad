import torch
import onnxruntime as ort
import numpy as np
from health_multimodal.image.utils import get_image_inference, ImageModelType

print("[AegisRad] Loading BioViL-T encoder...")
# 1. Initialize engine and extract the PyTorch model
engine = get_image_inference(ImageModelType.BIOVIL_T)
model = engine.model.encoder # Replaces the broken get_biovil_t_image_encoder() call

# 2. Set to evaluation mode
model.eval()

# 3. Create dummy input (Note: BioViL-T often uses larger resolutions like 480x480 in production, 
# but 224x224 is perfectly fine for tracing the ONNX graph!)
dummy = torch.randn(1, 3, 224, 224)

print("[AegisRad] Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy,
    "models/biovil_encoder.onnx",
    input_names=["image"],
    output_names=["features"],
    dynamic_axes={"image": {0: "batch"}},
    opset_version=14
)

print("[AegisRad] Validating export...")
sess = ort.InferenceSession("models/biovil_encoder.onnx")
out  = sess.run(None, {"image": dummy.numpy()})
print(f"Output shape: {out[0].shape}")

print("[AegisRad] Encoder export complete.")