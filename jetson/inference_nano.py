import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

# Add local path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from aegisrad.models import VisionEncoder, QFormer, ClinicalHead, ReflexiveProjector
from aegisrad.llm import GemmaLLM
from aegisrad.projector import Projector

class AegisRadNano:
    def __init__(self, 
                 ckpt_path="../Code Nano/models/best/components.pt",
                 llm_path="../Code Nano/models/gemma-2-2b-it-Q4_K_M.gguf"):
        
        # Determine best available device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        print(f"🚀 AegisRad Nano Initializing on: {self.device}")

        # 1. Load Pathology Stack
        self.encoder = VisionEncoder().to(self.device)
        self.qformer = QFormer().to(self.device)
        self.clinical_head = ClinicalHead(num_classes=14, use_attention=True).to(self.device)
        
        # Load weights
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['vision_encoder'])
        self.qformer.load_state_dict(checkpoint['qformer'])
        self.clinical_head.load_state_dict(checkpoint['clinical_head'])
        
        # 2. Load Projector Bridge
        self.projector_wrapper = Projector(state_dict=checkpoint['reflexive_proj'])
        
        # Set to eval
        self.encoder.eval()
        self.qformer.eval()
        self.clinical_head.eval()
        
        # 3. Load Gemma-2B
        self.llm = GemmaLLM(model_path=llm_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.conditions = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]

    def run_inference(self, image_input):
        """Processes an image and returns a dict with report and ailments."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Step A: Vision extraction
            features, _ = self.encoder(img_tensor)
            queries = self.qformer(features) 
            
            # Step B: Pathology Detection
            logits = self.clinical_head(queries)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Filter ailments (Only show name, no percentage as requested)
            flagged = {self.conditions[i]: float(probs[i]) for i in range(14) if probs[i] > 0.4}
            ailments = [c for c, p in flagged.items() if c != "No Finding"]
            
            # Step C: VLM Alignment
            visual_hints = self.projector_wrapper.project(queries)
            
            # Step D: Generative Report
            report_raw = self.llm.generate_report(visual_hints, flagged)
            
        # Parse report (split from internal metadata)
        main_report = report_raw.split("[Clinical Mode")[0].strip()
        
        # Extract Recommendation (Assuming LLM followed format)
        sections = main_report.split("RECOMMENDATION:")
        report_body = sections[0].strip()
        recommendation = sections[1].strip() if len(sections) > 1 else "Clinical correlation and follow-up as indicated."
        
        return {
            "report": report_body,
            "recommendation": recommendation,
            "ailments": ailments
        }

if __name__ == "__main__":
    # Test local run
    engine = AegisRadNano()
    # Replace with a real test path if needed
    print("Engine Ready.")
