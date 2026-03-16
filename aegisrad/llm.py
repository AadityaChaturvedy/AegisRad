import numpy as np
from llama_cpp import Llama
from config import GEMMA_GGUF, GEMMA_CTX_LEN, GEMMA_MAX_TOKENS, GEMMA_THREADS, GEMMA_GPU_LAYERS

TRIAGE_PROMPT = """<start_of_turn>user
You are a clinical radiologist AI. Write a structured radiology report.
Describe findings radiologically in clinical language.

Urgency rules:
- ROUTINE: No Finding only
- MODERATE: 1-2 non-critical findings under 60% confidence  
- URGENT: 3 or more findings, OR any finding above 60%, OR Pneumonia present
- CRITICAL: Pneumothorax above 70% or life-threatening finding above 80%

Flagged conditions:
{flagged_conditions}

Respond in exactly this format:
FINDINGS: [radiological observations in clinical language]
IMPRESSION: [one sentence interpretation]
SEVERITY: [1-5]
URGENCY: [ROUTINE or MODERATE or URGENT or CRITICAL]
RECOMMENDATION: [one sentence next step]
<end_of_turn>
<start_of_turn>model
FINDINGS:"""


class GemmaLLM:
    def __init__(self):
        self.llm = Llama(
            model_path=GEMMA_GGUF,
            n_ctx=GEMMA_CTX_LEN,
            n_threads=GEMMA_THREADS,
            n_gpu_layers=GEMMA_GPU_LAYERS,
            verbose=False
        )
        print("[AegisRad] Gemma-2B loaded.")

    def generate_report(self,
                        projected_features: np.ndarray,
                        flagged_conditions: dict) -> str:
        conditions_text = self._format_conditions(flagged_conditions)
        prompt = TRIAGE_PROMPT.format(
            flagged_conditions=conditions_text
        )
        output = self.llm(
            prompt,
            max_tokens=GEMMA_MAX_TOKENS,
            temperature=0.1,
            stop=["<end_of_turn>", "<start_of_turn>"]
        )
        return "FINDINGS:" + output["choices"][0]["text"]

    def _format_conditions(self, flagged: dict) -> str:
        if not flagged:
            return "No significant conditions detected."
        return "\n".join(
            f"  - {cond}: {prob:.1%} confidence"
            for cond, prob in sorted(
                flagged.items(), key=lambda x: x[1], reverse=True
            )
        )