import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import gc

# Add path for internal imports
sys.path.append("/Volumes/SSD/Jetson_AegisRad/Code Nano")
from aegisrad.models import ClinicalHead, ReflexiveProjector

# Config
CACHE_DIR = "/Volumes/SSD/Jetson_AegisRad/cache"
CKPT_PATH = "/Volumes/SSD/Jetson_AegisRad/Code Nano/models/best/components.pt"
OUTPUT_CKPT = "/Volumes/SSD/Jetson_AegisRad/Code Nano/models/best/components.pt"
MAX_SAMPLES = 50000
BATCH_SIZE = 512
EPOCHS = 10

class AlignmentDataset(Dataset):
    def __init__(self, cache_dir, max_samples=None):
        self.chunk_files = sorted([f for f in os.listdir(cache_dir) if f.startswith("train_chunk")])
        self.chunk_metadata = []
        self.total_len = 0
        
        print(f"📂 Scanning {len(self.chunk_files)} chunks for VLM alignment...")
        for i, cf in enumerate(self.chunk_files):
            path = os.path.join(cache_dir, cf)
            data = torch.load(path, map_location='cpu', weights_only=False)
            labels = data['labels'].clone()
            num_samples = labels.size(0)
            
            self.chunk_metadata.append({
                'path': path,
                'start_idx': self.total_len,
                'num_samples': num_samples,
                'labels': labels
            })
            self.total_len += num_samples
            del data
            if max_samples and self.total_len >= max_samples:
                self.total_len = max_samples
                break
            if i % 20 == 0: gc.collect()
            
        self.current_chunk_idx = -1
        self.current_features = None

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # find which chunk
        chunk_idx = -1
        for i, meta in enumerate(self.chunk_metadata):
            if idx >= meta['start_idx'] and idx < meta['start_idx'] + meta['num_samples']:
                chunk_idx = i
                break
        
        if chunk_idx != self.current_chunk_idx:
            path = self.chunk_metadata[chunk_idx]['path']
            data = torch.load(path, map_location='cpu', weights_only=False)
            self.current_features = data['features']
            self.current_chunk_idx = chunk_idx
            del data
            
        local_idx = idx - self.chunk_metadata[chunk_idx]['start_idx']
        feature = self.current_features[local_idx]
        label = self.chunk_metadata[chunk_idx]['labels'][local_idx]
        
        return feature, label

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 VLM Alignment Device: {device}")

    # 1. Load Models
    projector = ReflexiveProjector().to(device)
    # We use a clinical anchor to ensure semantic alignment
    clinical_head = ClinicalHead(num_classes=14, use_attention=True).to(device)
    
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if 'clinical_head' in checkpoint:
        clinical_head.load_state_dict(checkpoint['clinical_head'])
    if 'reflexive_proj' in checkpoint: # Load existing if available
        projector.load_state_dict(checkpoint['reflexive_proj'])
        
    clinical_head.eval() # Anchor is frozen
    projector.train()

    # 2. Dataset
    dataset = AlignmentDataset(CACHE_DIR, max_samples=MAX_SAMPLES)
    # Sequential loading is MUCH more memory efficient on 8GB RAM for this task
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 3. Optimization
    optimizer = optim.AdamW(projector.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    print("⚡ Starting Phase 2b: Reflexive Alignment (Memory-Optimized)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Align]")
        
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Pool the 32 queries into a single semantic summary for the projector
            pooled_features = features.mean(dim=1) 
            projected = projector(pooled_features)
            
            logits = clinical_head(projected)
            
            loss_anchor = criterion(logits, labels)
            std_loss = 1.0 - torch.std(projected, dim=1).mean()
            loss = loss_anchor + 0.1 * std_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Explicit memory management for 8GB RAM
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        print(f"Epoch {epoch+1:02d} Summary | Loss: {total_loss/len(dataloader):.4f}")

    # 4. Save
    print("🎉 Alignment Complete. Updating checkpoint...")
    checkpoint['reflexive_proj'] = projector.state_dict()
    torch.save(checkpoint, OUTPUT_CKPT)
    print("✅ components.pt updated with aligned ReflexiveProjector.")

if __name__ == "__main__":
    main()
