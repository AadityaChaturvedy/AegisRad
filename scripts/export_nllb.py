from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SAVE_PATH  = "models/nllb-200-distilled-600M"

print("[AegisRad] Downloading NLLB-200 distilled 600M...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"[AegisRad] Saving to {SAVE_PATH}...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print("[AegisRad] NLLB-200 download complete.")