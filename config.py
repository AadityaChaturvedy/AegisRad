import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
ENCODER_ONNX  = os.path.join(BASE_DIR, "models/biovil_encoder.onnx")
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
CLASS_THRESHOLDS = {
    'No Finding': 0.55, 'Enlarged Cardiomediastinum': 0.57, 'Cardiomegaly': 0.47,
    'Lung Opacity': 0.56, 'Lung Lesion': 0.5, 'Edema': 0.42,
    'Consolidation': 0.4, 'Pneumonia': 0.58, 'Atelectasis': 0.61,
    'Pneumothorax': 0.59, 'Pleural Effusion': 0.39, 'Pleural Other': 0.49,
    'Fracture': 0.44, 'Support Devices': 0.56
}

CONFIDENCE_THRESHOLD = 0.5 # Default fallback
CONDITIONS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# Supported languages
LANGUAGES = {
    "Acehnese (Arabic script)":  "ace_Arab",
    "Arabic":                    "arb_Arab",
    "Assamese":                  "asm_Beng",
    "Awadhi":                    "awa_Deva",
    "Banjara / Lambadi":         "bjj_Deva",
    "Bengali":                   "ben_Beng",
    "Bhojpuri":                  "bho_Deva",
    "Chhattisgarhi":             "hne_Deva",
    "Dogri":                     "dgo_Deva",
    "English":                   "eng_Latn",
    "French":                    "fra_Latn",
    "Gujarati":                  "guj_Gujr",
    "Hindi":                     "hin_Deva",
    "Kannada":                   "kan_Knda",
    "Kashmiri (Arabic script)":  "kas_Arab",
    "Kashmiri (Devanagari)":     "kas_Deva",
    "Konkani":                   "gom_Deva",
    "Maithili":                  "mai_Deva",
    "Malayalam":                 "mal_Mlym",
    "Manipuri (Bengali script)": "mni_Beng",
    "Manipuri (Meitei script)":  "mni_Mtei",
    "Marathi":                   "mar_Deva",
    "Mizo / Lushai":             "lus_Latn",
    "Nepali":                    "npi_Deva",
    "Odia":                      "ory_Orya",
    "Punjabi (Gurmukhi)":        "pan_Guru",
    "Portuguese":                "por_Latn",
    "Sanskrit":                  "san_Deva",
    "Santali":                   "sat_Olck",
    "Sindhi (Arabic script)":    "snd_Arab",
    "Sindhi (Devanagari)":       "snd_Deva",
    "Spanish":                   "spa_Latn",
    "Tamil":                     "tam_Taml",
    "Telugu":                    "tel_Telu",
    "Urdu":                      "urd_Arab",
}

# Inference
DEVICE = "cuda"
