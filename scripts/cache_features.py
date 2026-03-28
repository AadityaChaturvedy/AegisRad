import os
import sys
import torch
import pandas as pd
import ast
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Import architectures
sys.path.append("/Volumes/SSD/Jetson_AegisRad/Code Nano")
from aegisrad.models import VisionEncoder, QFormer

# Config
DATASET_DIR = "/Volumes/SSD/RRA/Dataset/mimic_micro_split"
CKPT_PATH = "/Volumes/SSD/Jetson_AegisRad/Code Nano/models/best/components.pt"
CACHE_DIR = "/Volumes/SSD/Jetson_AegisRad/cache"
CHUNK_SIZE = 1000
MAX_SAMPLES = 100000

IMAGE_SIZE = 224
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

class ExtractDataset(Dataset):
    def __init__(self, csv_file, images_dir, max_samples=None):
        self.data = pd.read_csv(csv_file)
        if max_samples and len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        ])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_path'])
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            success = True
        except Exception:
            img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            success = False
            
        label_str = row['clinical_labels']
        try:
            parsed = ast.literal_eval(label_str) if isinstance(label_str, str) else label_str
            labels = torch.tensor(parsed, dtype=torch.float32)
        except Exception:
            labels = torch.zeros(14)
            
        return img, labels, success

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Feature Extraction Device: {device}")

    # Load Models
    encoder = VisionEncoder().to(device)
    qformer = QFormer().to(device)
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['vision_encoder'])
    qformer.load_state_dict(checkpoint['qformer'])
    encoder.eval()
    qformer.eval()
    encoder.half() # Use fp16
    qformer.half()
    
    # Dataset
    dataset = ExtractDataset(
        os.path.join(DATASET_DIR, 'train.csv'), 
        os.path.join(DATASET_DIR, 'images'),
        max_samples=MAX_SAMPLES
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=2) # Reduced workers for safety

    os.makedirs(CACHE_DIR, exist_ok=True)
    
    current_chunk_features = []
    current_chunk_labels = []
    samples_in_current_chunk = 0
    chunk_idx = 0
    total_processed = 0

    with torch.no_grad():
        for images, labels, successes in tqdm(dataloader, desc="⚡ Caching 100k Features"):
            images = images.to(device).half()
            
            features, _ = encoder(images)
            queries = qformer(features) # [B, 32, 2048]
            
            current_chunk_features.append(queries.cpu().float())
            current_chunk_labels.append(labels)
            
            num_b = images.size(0)
            samples_in_current_chunk += num_b
            total_processed += num_b
            
            if samples_in_current_chunk >= CHUNK_SIZE:
                save_path = os.path.join(CACHE_DIR, f"train_chunk_{chunk_idx}.pt")
                torch.save({
                    'features': torch.cat(current_chunk_features, dim=0),
                    'labels': torch.cat(current_chunk_labels, dim=0)
                }, save_path)
                print(f"✅ Saved chunk {chunk_idx} ({samples_in_current_chunk} samples)")
                chunk_idx += 1
                current_chunk_features = []
                current_chunk_labels = []
                samples_in_current_chunk = 0

    # Save final remaining chunk
    if current_chunk_features:
        save_path = os.path.join(CACHE_DIR, f"train_chunk_{chunk_idx}.pt")
        torch.save({
            'features': torch.cat(current_chunk_features, dim=0),
            'labels': torch.cat(current_chunk_labels, dim=0)
        }, save_path)
        print(f"✅ Saved final chunk {chunk_idx}")

    print(f"✨ Feature extraction complete. Total: {total_processed}")

if __name__ == "__main__":
    main()
