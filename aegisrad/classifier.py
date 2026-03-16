import numpy as np
import onnxruntime as ort
from config import CONDITIONS

class Classifier:
    def __init__(self):
        self.session = ort.InferenceSession(
            "models/clinical_head.onnx",
            providers=["CPUExecutionProvider"]
        )
        print("[AegisRad] Classification head loaded.")

    def predict(self, pooled: np.ndarray) -> dict:
        # pooled: [1, 2048] from encoder
        text_repr = np.zeros_like(pooled)   # zero text at inference
        logits    = self.session.run(
            None,
            {
                "visual_pooled": pooled,
                "text_repr":     text_repr
            }
        )[0]
        probs = 1 / (1 + np.exp(-logits))   # sigmoid
        return {
            cond: float(round(prob, 4))
            for cond, prob in zip(CONDITIONS, probs.squeeze())
        }