import gc
import time
import os
import torch
import numpy as np
import onnxruntime as ort
from aegisrad.preprocessor     import Preprocessor
from aegisrad.encoder          import BioViLEncoder
from aegisrad.classifier       import Classifier
from aegisrad.qadapter         import RRAQAdapter
from aegisrad.projector        import Projector
from aegisrad.llm              import GemmaLLM
from aegisrad.formatter        import ReportFormatter
from aegisrad.threshold_filter import ThresholdFilter
from aegisrad.translator       import NLLBTranslator
from config import LANGUAGES, ENCODER_ONNX, QFORMER_ONNX

_CKPT_PATH = 'models/best/components.pt'

class AegisRadPipeline:
    def __init__(self, use_onnx=True):
        print("[AegisRad] Initializing optimized pipeline...")
        self.use_onnx = use_onnx
        
        # 1. Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Note: CoreMLExecutionProvider is better for Mac if available
            self.device = "mps"
            self.ort_providers = ["CPUExecutionProvider"]
        else:
            self.device = "cpu"
            self.ort_providers = ["CPUExecutionProvider"]
            
        print(f"[AegisRad] Target device: {self.device}")

        # 2. Check for ONNX models if requested
        self.encoder_session = None
        self.qadapter_session = None
        
        if self.use_onnx:
            if os.path.exists(ENCODER_ONNX) and os.path.exists(QFORMER_ONNX):
                try:
                    print("[AegisRad] Loading ONNX engines for high-speed inference...")
                    self.encoder_session = ort.InferenceSession(ENCODER_ONNX, providers=self.ort_providers)
                    self.qadapter_session = ort.InferenceSession(QFORMER_ONNX, providers=self.ort_providers)
                    print("✅ ONNX sessions ready.")
                except Exception as e:
                    print(f"⚠️ ONNX load failed, falling back to PyTorch: {e}")
                    self.encoder_session = None

        # 3. Load checkpoint for remaining/fallback components
        print("[AegisRad] Loading checkpoint...")
        checkpoint = torch.load(_CKPT_PATH, map_location='cpu', weights_only=False)

        self.preprocessor = Preprocessor()
        
        # Vision Encoder (PyTorch fallback)
        if not self.encoder_session:
            self.encoder = BioViLEncoder(state_dict=checkpoint.pop('vision_encoder'))
        else:
            _ = checkpoint.pop('vision_encoder', None)
            self.encoder = None

        # Clinical Head (Always PyTorch for flexibility)
        self.classifier = Classifier(state_dict=checkpoint.pop('clinical_head'))

        # Q-Adapter (PyTorch fallback)
        if not self.qadapter_session:
            self.qadapter = RRAQAdapter(state_dict=checkpoint.pop('qformer'))
        else:
            _ = checkpoint.pop('qformer', None)
            self.qadapter = None

        # Projector (PyTorch)
        self.projector = Projector(state_dict=checkpoint.pop('reflex_proj'))

        del checkpoint
        gc.collect()

        self.llm        = GemmaLLM()
        self.formatter  = ReportFormatter()
        self.filter     = ThresholdFilter()
        self.translator = NLLBTranslator()
        print("[AegisRad] All components ready.")

    def run(self, image_path: str, language: str = "English") -> dict:
        t0 = time.time()
        timings = {}

        # Stage 1 — Preprocessing
        ts = time.time()
        image = self.preprocessor.process(image_path)
        timings['preprocess'] = time.time() - ts

        with torch.no_grad():
            # Stage 2 — Vision Encoding
            ts = time.time()
            if self.encoder_session:
                # ONNX path
                outputs = self.encoder_session.run(None, {"image": image})
                features = torch.from_numpy(outputs[0]).to(self.device).half()
                # Need pooled version for classifier
                pooled = features.mean(dim=[2, 3]) 
            else:
                # PyTorch path
                features, pooled = self.encoder.encode(image)
            timings['encode'] = time.time() - ts
            del image

            # Stage 3 — Clinical Classification
            ts = time.time()
            all_probs = self.classifier.predict(pooled)
            flagged   = self.filter.filter(all_probs)
            timings['classify'] = time.time() - ts

            # Stage 4 — Q-Adapter
            ts = time.time()
            if self.qadapter_session:
                # ONNX path expects [B, 2048, 7, 7]
                # BioViL-T features are already [B, 2048, 7, 7]
                outputs = self.qadapter_session.run(None, {"visual_features": features.float().cpu().numpy()})
                queries = torch.from_numpy(outputs[0]).to(self.device).half()
            else:
                queries = self.qadapter.extract_queries(features)
            timings['qadapter'] = time.time() - ts
            del features
            del pooled

            # Stage 5 — Projector
            ts = time.time()
            projected = self.projector.project(queries)
            timings['project'] = time.time() - ts
            del queries

        # Stage 6 — LLM Report (Rule-based)
        ts = time.time()
        raw_text = self.llm.generate_report(projected, flagged)
        timings['llm'] = time.time() - ts
        del projected

        # Stage 7 — Formatting & Translation
        ts = time.time()
        report = self.formatter.parse(raw_text)
        report = self.translator.translate_report(report, language)
        timings['format_translate'] = time.time() - ts

        # Metadata
        report["condition_probs"] = all_probs
        report["flagged"]         = flagged
        report["language"]        = language
        report["latency_s"]       = round(time.time() - t0, 3)
        report["timings"]         = {k: round(v, 4) for k, v in timings.items()}
        report["raw_output"]      = raw_text

        print(f"[AegisRad] Done | {report['latency_s']}s | {language} | "
              f"Severity {report['severity_score']}/5")

        return report
