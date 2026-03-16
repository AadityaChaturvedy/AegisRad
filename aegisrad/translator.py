import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import NLLB_MODEL_DIR, NLLB_MAX_TOKENS, LANGUAGES


class NLLBTranslator:
    def __init__(self):
        print("[AegisRad] Loading NLLB-200 translator...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_DIR)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL_DIR,
            dtype=torch.float16
        )
        self.model.eval()
        print("[AegisRad] NLLB-200 loaded.")

    def translate(self, text: str, target_language: str) -> str:
        target_code = LANGUAGES.get(target_language, "eng_Latn")

        if target_code == "eng_Latn":
            return text

        self.tokenizer.src_lang = "eng_Latn"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get target language token id correctly
        target_lang_id = self.tokenizer.convert_tokens_to_ids(target_code)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=NLLB_MAX_TOKENS
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    def translate_report(self, report: dict, language: str) -> dict:
        if LANGUAGES.get(language, "eng_Latn") == "eng_Latn":
            return report

        fields_to_translate = ["findings", "impression", "recommendation"]
        for field in fields_to_translate:
            if field in report and report[field] != "Not determined":
                report[field] = self.translate(report[field], language)

        return report