import time
import numpy as np
from aegisrad.preprocessor     import Preprocessor
from aegisrad.encoder          import BioViLEncoder
from aegisrad.classifier       import Classifier
from aegisrad.qadapter         import RRAQAdapter
from aegisrad.projector        import Projector
from aegisrad.llm              import GemmaLLM
from aegisrad.formatter        import ReportFormatter
from aegisrad.threshold_filter import ThresholdFilter
from aegisrad.translator       import NLLBTranslator
from config import LANGUAGES


class AegisRadPipeline:
    def __init__(self):
        print("[AegisRad] Initializing pipeline...")
        self.preprocessor = Preprocessor()
        self.encoder      = BioViLEncoder()
        self.classifier   = Classifier()
        self.qadapter     = RRAQAdapter()
        self.projector    = Projector()
        self.llm          = GemmaLLM()
        self.formatter    = ReportFormatter()
        self.filter       = ThresholdFilter()
        self.translator   = NLLBTranslator()
        print("[AegisRad] All components ready.")

    def run(self, image_path: str, language: str = "English") -> dict:
        t0 = time.time()

        # Stage 1 — Preprocessing
        image = self.preprocessor.process(image_path)

        # Stage 2 — BioViL-T visual encoding
        features, pooled  = self.encoder.encode(image)

        # Stage 3 — Classification head
        # Gets per-condition probabilities from BioViL-T features
        all_probs = self.classifier.predict(pooled)
        flagged   = self.filter.filter(all_probs)

        # Stage 4 — RRA-Q grounded query extraction
        queries   = self.qadapter.extract_queries(features)

        # Stage 5 — ReflexiveProjector
        projected = self.projector.project(queries)

        # Stage 6 — Llama report generation
        # Conditioned on both visual features AND flagged conditions
        raw_text  = self.llm.generate_report(projected, flagged)

        # Stage 7 — Structure the report
        report    = self.formatter.parse(raw_text)

        # Stage 8 — Translate if needed
        report    = self.translator.translate_report(report, language)

        # Attach metadata
        report["condition_probs"] = all_probs
        report["flagged"]         = flagged
        report["language"]        = language
        report["latency_s"]       = round(time.time() - t0, 2)
        report["raw_output"]      = raw_text

        print(f"[AegisRad] Inference complete — "
              f"{report['latency_s']}s | {language} | "
              f"Severity {report['severity_score']}/5")

        return report