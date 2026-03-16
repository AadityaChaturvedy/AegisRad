import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
ENCODER_TRT  = os.path.join(BASE_DIR, "models/biovil_encoder.trt")
CLASSIFIER_PT = os.path.join(BASE_DIR, "models/biovil_classifier.pt")
QFORMER_ONNX = os.path.join(BASE_DIR, "models/rra_qadapter.onnx")
PROJECTOR_PT = os.path.join(BASE_DIR, "models/projector.pt")
NLLB_MODEL_DIR = os.path.join(BASE_DIR, "models/nllb-200-distilled-600M")

GEMMA_GGUF = os.path.join(BASE_DIR, "models/gemma-2-2b-it-Q4_K_M.gguf")
LORA_DIR   = os.path.join(BASE_DIR, "models/lora")

# Preprocessing
IMAGE_SIZE = 224
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD  = [0.229, 0.224, 0.225]

# Q-Adapter
NUM_QUERY_TOKENS = 32
QUERY_DIM        = 768

# Llama
GEMMA_CTX_LEN    = 2048
GEMMA_MAX_TOKENS = 400
GEMMA_THREADS    = 4
GEMMA_GPU_LAYERS = 20

# Translation
NLLB_MAX_TOKENS = 256

# Classification
CONFIDENCE_THRESHOLD = 0.5
CONDITIONS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# Supported languages
LANGUAGES = {
    "English":    "eng_Latn",
    "Tamil":      "tam_Taml",
    "Hindi":      "hin_Deva",
    "French":     "fra_Latn",
    "Arabic":     "arb_Arab",
    "Spanish":    "spa_Latn",
    "Portuguese": "por_Latn",
    "Bengali":    "ben_Beng",
    "Telugu":     "tel_Telu",
    "Kannada":    "kan_Knda"
}

# Inference
DEVICE = "cuda"