# Models Directory

This directory stores model weights, TensorRT engines, ONNX graphs, and quantized models required for AegisRad runtime inference.

## Required Models

Place or export the following models into this directory:

| Filename | Description | Export / Download Script |
|---|---|---|
| `biovil_encoder.onnx` / `biovil_encoder.trt` | BioViL-T ResNet-50 visual feature extractor | `python scripts/export_encoder.py` |
| `clinical_head.onnx` / `biovil_classifier.pt` | 14-class CheXpert pathology classifier head | `python scripts/init_weights.py` |
| `rra_qadapter.onnx` | Reflexive-Radio-Adapter Q-Former compression module | `python scripts/export_qadapter.py` |
| `projector.pt` | Multi-modal alignment projection layer | `python scripts/init_weights.py` |
| `gemma-2-2b-it-Q4_K_M.gguf` | Gemma-2B-IT GGUF 4-bit quantized triage LLM | Download from HuggingFace (`llama.cpp` compatible) |
| `nllb-200-distilled-600M/` | Meta NLLB-200 distilled translation model | `python scripts/export_nllb.py` |

## Model Export Utilities

Run the provided helper scripts in `scripts/`:
```bash
# 1. Initialize classifier & projector weights
python scripts/init_weights.py

# 2. Export BioViL-T encoder to ONNX
python scripts/export_encoder.py

# 3. Export RRA-Q Adapter to ONNX
python scripts/export_qadapter.py

# 4. Download and prepare NLLB-200 translation model
python scripts/export_nllb.py
```
