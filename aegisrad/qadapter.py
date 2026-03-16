import numpy as np
import platform
from config import QFORMER_ONNX

IS_JETSON = platform.machine() == "aarch64"
import onnxruntime as ort


class RRAQAdapter:
    """
    RRA-Q Adapter Module
    Custom Q-Former variant with ungrounded token penalization.
    Trained on MIMIC-CXR via RRA pipeline.
    """
    def __init__(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if IS_JETSON else
            ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            "models/rra_qadapter.onnx",
            providers=providers
        )
        print("[AegisRad] RRA-Q Adapter loaded.")

    def extract_queries(self, visual_features: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            None,
            {"visual_features": visual_features}
        )
        return outputs[0]   # [1, 32, 2048]