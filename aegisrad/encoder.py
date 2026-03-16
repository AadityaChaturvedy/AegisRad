import numpy as np
import platform
from config import ENCODER_TRT

IS_JETSON = platform.machine() == "aarch64"

if IS_JETSON:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
else:
    import onnxruntime as ort


class BioViLEncoder:
    def __init__(self):
        if IS_JETSON:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(ENCODER_TRT, "rb") as f:
                runtime     = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.mode    = "trt"
            print("[AegisRad] BioViL-T encoder loaded (TensorRT).")
        else:
            self.session = ort.InferenceSession(
                "models/biovil_encoder.onnx",
                providers=["CPUExecutionProvider"]
            )
            self.mode = "onnx"
            print("[AegisRad] BioViL-T encoder loaded (ONNX).")

    def encode(self, image: np.ndarray):
        if self.mode == "onnx":
            outputs  = self.session.run(None, {"image": image})
            features = outputs[0]   # [1, 2048, 7, 7]
            pooled   = outputs[1]   # [1, 2048]
            return features, pooled

        # TensorRT path (Jetson only)
        input_mem  = cuda.mem_alloc(image.nbytes)
        features   = np.empty((1, 2048, 7, 7), dtype=np.float32)
        pooled     = np.empty((1, 2048), dtype=np.float32)
        feat_mem   = cuda.mem_alloc(features.nbytes)
        pool_mem   = cuda.mem_alloc(pooled.nbytes)

        cuda.memcpy_htod(input_mem, image)
        self.context.execute_v2([int(input_mem), int(feat_mem), int(pool_mem)])
        cuda.memcpy_dtoh(features, feat_mem)
        cuda.memcpy_dtoh(pooled,   pool_mem)
        return features, pooled