# AegisRad: Edge Deployment of Joint Chest X-Ray Classification and Multilingual Report Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Target: NVIDIA Jetson](https://img.shields.io/badge/Target-NVIDIA%20Jetson%20Nano%20%2F%20Orin-green.svg)](https://developer.nvidia.com/embedded-computing)
[![Inference: ONNX / TensorRT](https://img.shields.io/badge/Inference-ONNX%20%7C%20TensorRT%20%7C%20GGUF-orange.svg)](https://onnxruntime.ai/)

**AegisRad** is an edge-deployed multimodal radiological intelligence system designed for automated Chest X-Ray (CXR) pathology triage, multi-label classification, and structured multilingual clinical report generation. Engineered specifically to run entirely on-device (e.g., NVIDIA Jetson Nano / Orin Nano with <= 4 GB unified memory) with zero cloud dependencies, AegisRad brings hospital-grade radiological diagnostic support to remote clinics and low-resource healthcare settings.

---

## Key Features

- **Joint Classification and Reporting**: Simultaneously predicts 14 CheXpert pathology classes and generates full clinical radiology reports (Findings, Impression, Severity, Urgency, Recommendation).
- **Edge-Optimized Multi-stage Pipeline**: Operates within a 2.6 GB peak memory footprint on NVIDIA Jetson embedded hardware with sequential stage execution and explicit memory deallocation.
- **BioViL-T Vision Backbone**: ResNet-50 visual feature extractor optimized with ONNX Runtime and TensorRT INT8/FP16 acceleration.
- **Reflexive-Radio-Adapter (RRA-Q)**: Cross-attention Q-Former variant that compresses 49 spatial patch representations into 32 grounded query tokens with ungrounded token penalization.
- **4-bit Quantized Gemma-2B-IT**: Lightweight LLM fine-tuned with LoRA (Low-Rank Adaptation) running via `llama.cpp` / `llama-cpp-python` GGUF (Q4_K_M).
- **35+ Regional and Global Languages**: Integrated on-device translation using Meta NLLB-200 distilled 600M for instant report translation into regional Indian (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, etc.) and international languages.
- **Automated Clinical Triage Layer**: Rule-based priority gating computing 5-tier Severity Scores (1-5) and 4-tier Urgency indicators (Routine, Moderate, Urgent, Critical).
- **Interactive Gradio Web Interface**: Web interface for uploading X-ray images, selecting target languages, viewing confidence charts, and generating formatted clinical reports.

---

## Architecture and Pipeline Overview

```
                          +------------------------+
                          |   Chest X-Ray Image    |
                          +-----------+------------+
                                      |
                         [Stage 1: Preprocessor]
                                      |
                         [Stage 2: BioViL-T Encoder]
                             (TensorRT / ONNX)
                                     / \
                                    /   \
  [Stage 3: Multi-Label Classifier]      [Stage 4: RRA-Q Adapter]
     (14 CheXpert Pathologies)             (32 Query Tokens)
                |                                    |
   [Stage 3b: Threshold Filter]          [Stage 5: Projector]
                |                                    |
                +--------------+---------------------+
                               |
               [Stage 6: Gemma-2B LLM (GGUF Q4_K_M)]
                               |
               [Stage 7: ReportFormatter & Parser]
                               |
               [Stage 8: NLLB-200 Multilingual Translator]
                               |
                               v
               +---------------------------------+
               |   Structured Clinical Report    |
               |  * Findings & Impression        |
               |  * Severity Score (1-5)         |
               |  * Urgency Flag                 |
               |  * Next Step Recommendation     |
               |  * Localized Language Output    |
               +---------------------------------+
```

---

## Repository Structure

```
Final_AegisRad/
|-- aegisrad/                      # Core AegisRad Python package
|   |-- __init__.py                # Package initializer
|   |-- classifier.py              # 14-class CheXpert pathology classifier head
|   |-- encoder.py                 # BioViL-T visual encoder (ONNX / TensorRT / PyTorch)
|   |-- formatter.py               # Radiology report regex parser & triage formatter
|   |-- llm.py                     # Gemma-2B-IT GGUF & Fallback report generator
|   |-- models.py                  # PyTorch model definitions (QFormer, Projector, Classifier)
|   |-- preprocessor.py            # Image normalization & resizing pipeline
|   |-- projector.py               # Multimodal alignment projector
|   |-- qadapter.py                # RRA-Q query adapter module
|   |-- threshold_filter.py        # Calibrated per-condition threshold gating
|   `-- translator.py              # NLLB-200 distilled 600M multilingual translator
|-- assets/                        # Architecture diagrams, confusion matrices & UI assets
|   |-- Architecture_Diagram.pdf   # System architecture diagram (PDF)
|   |-- Architecture_Diagram.png   # System architecture diagram (PNG)
|   |-- cosine_confusion_matrix.png# Benchmark confusion matrix plot
|   |-- metrics_bar_graph.png      # Condition F1 & AUC-ROC metrics bar chart
|   `-- FrontendImage.jpeg         # Physical setup and UI preview image
|-- benchmarks/                    # Benchmark calculation suite & evaluation data
|   |-- compute_benchmarks_enhanced.py # Per-condition AUC-ROC & F1 computation
|   |-- metrics_utils.py           # Evaluation metric computation utilities
|   `-- bench_results_enhanced.json# Benchmark evaluation data
|-- jetson/                        # Dedicated NVIDIA Jetson Nano edge deployment suite
|   |-- inference_nano.py          # Optimized Jetson Nano edge runtime script
|   |-- run_on_nano.sh             # Shell execution script for Jetson
|   |-- install.sh                 # Jetson automated environment setup
|   |-- README_NANO.md             # Edge device setup guide
|   `-- app.py                     # Lightweight web UI for Jetson
|-- models/                        # Local model weights & ONNX/TRT engines (gitignored)
|   |-- README.md                  # Model download & setup instructions
|   `-- .gitkeep
|-- scripts/                       # Training, feature extraction & model export scripts
|   |-- train_mac_head.py          # Clinical classifier training script
|   |-- train_vlm_alignment.py     # Vision-language alignment fine-tuning
|   |-- eval_reports.py            # Clinical report evaluation utility
|   |-- cache_features.py          # Feature token pre-caching script
|   |-- export_encoder.py          # BioViL-T ONNX exporter
|   |-- export_nllb.py             # NLLB-200 translation model downloader
|   |-- export_qadapter.py         # RRA-Q Adapter ONNX exporter
|   `-- init_weights.py            # Head & projector weight initializer
|-- clinical_head_thresholds.json  # Calibrated classification decision thresholds
|-- config.py                      # Global configuration, model paths, labels & thresholds
|-- pipeline.py                    # AegisRadPipeline dual ONNX/PyTorch orchestrator
|-- requirements.txt               # Python package dependencies
|-- app.py                         # Gradio UI application server
|-- LICENSE                        # MIT License
`-- README.md                      # Project documentation
```

---

## Getting Started

### 1. Prerequisites

- **Python**: 3.10 or higher
- **Hardware**: NVIDIA Jetson Nano / Orin Nano / Xavier or any Linux/macOS/Windows workstation with CPU/CUDA support
- **C++ Build Tools**: Required for `llama-cpp-python` compilation

### 2. Installation

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/AadityaChaturvedy/AegisRad.git
cd AegisRad

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

*For Jetson Nano / JetPack 4.6+:*
Install NVIDIA JetPack libraries (`TensorRT`, `PyCUDA`, `ONNX Runtime GPU`).

### 3. Model Weights Initialization

Initialize model checkpoints and download necessary weights:

```bash
# 1. Initialize classification head and projector weights
python scripts/init_weights.py

# 2. Export BioViL-T encoder to ONNX format
python scripts/export_encoder.py

# 3. Export RRA-Q adapter module
python scripts/export_qadapter.py

# 4. Download and setup NLLB-200 translation model
python scripts/export_nllb.py
```

*Place `gemma-2-2b-it-Q4_K_M.gguf` inside the `models/` directory.*

---

## Usage

### Run the Interactive Web UI

Launch the Gradio web dashboard:

```bash
python app.py
```

Open your browser at `http://localhost:7860` to access the clinical triage portal.

### Python API Integration

```python
from pipeline import AegisRadPipeline

# Initialize the 7-stage pipeline
pipeline = AegisRadPipeline()

# Run end-to-end inference
report = pipeline.run("path/to/chest_xray.jpg", language="Hindi")

print("Severity:", report["severity_score"], f"({report['severity_label']})")
print("Urgency:", report["urgency"])
print("\n--- FINDINGS ---")
print(report["findings"])
print("\n--- IMPRESSION ---")
print(report["impression"])
print("\n--- RECOMMENDATION ---")
print(report["recommendation"])
```

---

## Supported Clinical Conditions

AegisRad evaluates **14 CheXpert pathology classes**:

1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

---

## Supported Languages (NLLB-200)

| Region / Language | Code | Region / Language | Code |
|---|---|---|---|
| English | `eng_Latn` | Hindi | `hin_Deva` |
| Tamil | `tam_Taml` | Telugu | `tel_Telu` |
| Bengali | `ben_Beng` | Marathi | `mar_Deva` |
| Gujarati | `guj_Gujr` | Kannada | `kan_Knda` |
| Malayalam | `mal_Mlym` | Punjabi | `pan_Guru` |
| Odia | `ory_Orya` | Assamese | `asm_Beng` |
| Urdu | `urd_Arab` | Nepali | `npi_Deva` |
| French | `fra_Latn` | Spanish | `spa_Latn` |

*(35+ dialects and regional Indian scripts supported, see [config.py](config.py))*

---

## Performance Benchmarks

| Metric | RRA / AegisRad Model |
|---|---|
| **Macro AUC-ROC** | **0.974 +- 0.003** |
| **Macro F1 Score** | **0.825 +- 0.007** |
| **CheXBERT Semantic Similarity** | **0.890 +- 0.004** |
| **Training Sample Efficiency** | 10,000 MIMIC-CXR studies |
| **Peak Memory on Jetson Nano** | **2.6 GB** (out of 4.0 GB) |
| **Target Hardware Cost** | < USD 150 |

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
