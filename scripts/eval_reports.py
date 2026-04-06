import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
import ast

# Add path for internal imports
sys.path.append("/Volumes/SSD/Jetson_AegisRad/Code Nano")
from aegisrad.models import VisionEncoder, QFormer, ClinicalHead
from aegisrad.llm import GemmaLLM
from aegisrad.projector import Projector

# Config
TEST_CSV = "/Volumes/SSD/RRA/Dataset/mimic_micro_split/test.csv"
IMAGES_DIR = "/Volumes/SSD/RRA/Dataset/mimic_micro_split/images"
CKPT_PATH = "/Volumes/SSD/Jetson_AegisRad/Code Nano/models/best/components.pt"
NUM_SAMPLES = 50

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Evaluation Device: {device}")

    # 1. Load Pathology Stack
    encoder = VisionEncoder().to(device)
    qformer = QFormer().to(device)
    clinical_head = ClinicalHead(num_classes=14, use_attention=True).to(device)
    
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['vision_encoder'])
    qformer.load_state_dict(checkpoint['qformer'])
    clinical_head.load_state_dict(checkpoint['clinical_head'])
    
    encoder.eval()
    qformer.eval()
    clinical_head.eval()

    # 2. Load LLM & Projector
    print("📀 Loading Gemma-2B VLM Engine & Aligned Projector...")
    llm = GemmaLLM()
    # Use the synchronized state dict
    projector_wrapper = Projector(state_dict=checkpoint['reflexive_proj'])

    # 3. Load Test Data
    df = pd.read_csv(TEST_CSV).sample(n=NUM_SAMPLES, random_state=42)
    
    # 4. Initialize Metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    rouge_scores = []
    bleu_scores = []

    print(f"🧐 Evaluating {NUM_SAMPLES} Image-to-Report generations...")

    for _, row in tqdm(df.iterrows(), total=NUM_SAMPLES):
        img_path = os.path.join(IMAGES_DIR, row['image_path'])
        ground_truth = str(row['report'])
        
        try:
            # Step A: Vision Pipeline
            from PIL import Image
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features, _ = encoder(img_tensor)
                queries = qformer(features) # [1, 32, 2048]
                logits = clinical_head(queries)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Step B: Filter pathologies (threshold 0.5)
            from aegisrad.llm import _CRITICAL, _MODERATE
            # Pathologies list matching training
            CONDITIONS = [
                'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                'Pleural Other', 'Fracture', 'Support Devices'
            ]
            flagged = {CONDITIONS[i]: float(probs[i]) for i in range(14) if probs[i] > 0.4}
            
            # Step C: Align & Generate VLM Report
            visual_hints = projector_wrapper.project(queries)
            generated = llm.generate_report(visual_hints, flagged)
            
            # Separate the generated report from the metadata
            report_text = generated.split("[Clinical Mode")[0].strip()

            # Step D: Scoring
            # ROUGE-L
            rs = scorer.score(ground_truth, report_text)
            rouge_scores.append(rs['rougeL'].fmeasure)
            
            # BLEU-4
            ref = ground_truth.split()
            hyp = report_text.split()
            bs = sentence_bleu([ref], hyp, smoothing_function=smoothing)
            bleu_scores.append(bs)
            
        except Exception as e:
            print(f"⚠️ Error processing {row['image_path']}: {e}")
            continue

    # Summary
    print("\n📊 --- AegisRad SOTA Metrics Summary ---")
    print(f"   Avg ROUGE-L: {np.mean(rouge_scores):.4f}")
    print(f"   Avg BLEU-4:  {np.mean(bleu_scores):.4f}")
    print(f"   Total Success: {len(rouge_scores)}/{NUM_SAMPLES}")
    print("------------------------------------------")

if __name__ == "__main__":
    main()
