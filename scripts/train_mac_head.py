import os
import sys
import ast
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import json
import csv
import sys

# Logger to mirror stdout to a file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Import architectures from Code Nano
from aegisrad.models import VisionEncoder, QFormer, ClinicalHead

# Config
DATASET_DIR = "/Volumes/SSD/RRA/Dataset/mimic_micro_split"
CKPT_PATH = "/Volumes/SSD/Jetson_AegisRad/Code Nano/models/best/components.pt"
OUTPUT_CKPT = "/Volumes/SSD/Jetson_AegisRad/Code Nano/models/best/components.pt"
THRESH_PATH = "/Volumes/SSD/Jetson_AegisRad/Code Nano/clinical_head_thresholds.json"

# Medical conditions mapping
CONDITIONS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

# Standardize image processing (matching training implementation exactly)
MAX_SAMPLES = 50000
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

class CachedMimicDataset(Dataset):
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
        except Exception as e:
            img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            
        label_str = row['clinical_labels']
        try:
            if isinstance(label_str, str):
                parsed = ast.literal_eval(label_str)
            else:
                parsed = label_str
            labels = torch.tensor(parsed, dtype=torch.float32)
            if labels.shape[0] != 14:
                labels = torch.zeros(14, dtype=torch.float32)
        except Exception:
            labels = torch.zeros(14, dtype=torch.float32)
            
        return img, labels

def asymmetric_focal_loss(logits, targets, pos_weights, gamma_pos=1, gamma_neg=2):
    p = torch.sigmoid(logits)
    loss_pos = -(1 - p)**gamma_pos * targets * torch.log(p + 1e-8) * pos_weights
    loss_neg = -p**gamma_neg * (1 - targets) * torch.log(1 - p + 1e-8)
    return (loss_pos + loss_neg).mean()

CACHE_DIR = "/Volumes/SSD/Jetson_AegisRad/cache"

import gc

class DiskFeatureDataset(Dataset):
    def __init__(self, cache_dir, max_samples=None):
        self.chunk_files = sorted([f for f in os.listdir(cache_dir) if f.startswith("train_chunk")])
        self.chunk_metadata = []
        self.total_len = 0
        self.max_samples = max_samples
        
        print(f"📂 Scanning {len(self.chunk_files)} chunks...")
        for i, cf in enumerate(self.chunk_files):
            path = os.path.join(cache_dir, cf)
            # Use weights_only=False for .pt files with dicts
            data = torch.load(path, map_location='cpu', weights_only=False)
            
            # IMPORTANT: Clone labels to detach them from the large feature storage
            labels = data['labels'].clone()
            num_samples = labels.size(0)
            
            self.chunk_metadata.append({
                'path': path,
                'start_idx': self.total_len,
                'num_samples': num_samples,
                'labels': labels
            })
            self.total_len += num_samples
            
            # Stop if we hit max_samples
            if self.max_samples and self.total_len >= self.max_samples:
                # Truncate the last chunk's labels if necessary
                over_limit = self.total_len - self.max_samples
                if over_limit > 0:
                    self.chunk_metadata[-1]['num_samples'] -= over_limit
                    self.chunk_metadata[-1]['labels'] = self.chunk_metadata[-1]['labels'][:self.chunk_metadata[-1]['num_samples']]
                self.total_len = self.max_samples
                del data
                break

            # Immediate cleanup
            del data
            if i % 20 == 0:
                gc.collect()
        
        self.current_chunk_idx = -1
        self.current_features = None
        self.current_labels = None

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # global_idx is passed directly when using Subset
        global_idx = idx
        
        # Find which chunk contains this global index
        for i, meta in enumerate(self.chunk_metadata):
            if meta['start_idx'] <= global_idx < (meta['start_idx'] + meta['num_samples']):
                if self.current_chunk_idx != i:
                    # Clear previous chunk and FORCE garbage collection for 8GB RAM
                    self.current_features = None
                    self.current_labels = None
                    gc.collect()
                    
                    # Load features and labels for this chunk
                    data = torch.load(meta['path'], map_location='cpu', weights_only=False)
                    self.current_features = data['features']
                    self.current_labels = data['labels']
                    self.current_chunk_idx = i
                
                local_idx = global_idx - meta['start_idx']
                return self.current_features[local_idx], self.current_labels[local_idx]
        
        raise IndexError(f"Global index {global_idx} out of range [0, {self.total_len})")

# For efficient training, we want to iterate chunk by chunk
# We'll create a simple generator or just sort the indices by chunk
def get_chunk_optimized_loader(dataset, batch_size, shuffle=True):
    # Sort indices by their chunk membership to minimize disk reads
    indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    metadata = base_dataset.chunk_metadata
    
    # Group indices by chunk using global indices
    chunk_groups = [[] for _ in range(len(metadata))]
    for idx in indices:
        for i, meta in enumerate(metadata):
            if meta['start_idx'] <= idx < (meta['start_idx'] + meta['num_samples']):
                chunk_groups[i].append(idx)
                break
    
    # Shuffle chunks if needed
    if shuffle:
        import random
        random.shuffle(chunk_groups)
        # Shuffle within each chunk group
        for group in chunk_groups:
            random.shuffle(group)
    
    # Flatten back into a single list of global indices
    optimized_indices = [idx for group in chunk_groups for idx in group]
    
    # Create an optimized subset
    optimized_subset = torch.utils.data.Subset(base_dataset, optimized_indices)
    
    return DataLoader(optimized_subset, batch_size=batch_size, shuffle=False)

def optimize_thresholds(targets, probs):
    """Find the best F1 threshold for each class independently."""
    num_classes = targets.shape[1]
    best_thresholds = np.zeros(num_classes)
    for i in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(targets[:, i], probs[:, i])
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
        best_thresholds[i] = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    return best_thresholds

def main():
    # Redirect stdout to log file
    sys.stdout = Logger("/Volumes/SSD/Jetson_AegisRad/training_log.txt")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Using device: {device}", flush=True)

    HISTORY_PATH = "/Volumes/SSD/Jetson_AegisRad/training_history.csv"
    SUMMARY_PATH = "/Volumes/SSD/Jetson_AegisRad/training_summary.json"

    # 1. Load clinical head (other models are for caching)
    clinical_head = ClinicalHead(num_classes=14, use_attention=True).to(device)
    
    # Check if cache exists
    if not os.path.exists(CACHE_DIR) or not any(f.startswith("train_chunk") for f in os.listdir(CACHE_DIR)):
        print("❌ Cache not found! Please run cache_features.py first.", flush=True)
        return

    # Load existing weights for Warmstart
    print("🔋 Warmstarting from previous best weights...", flush=True)
    checkpoint = torch.load(OUTPUT_CKPT, map_location=device, weights_only=False)
    if 'clinical_head' in checkpoint:
        clinical_head.load_state_dict(checkpoint['clinical_head'])
        print("✅ ClinicalHead weights loaded successfully.")

    # 2. Load Datasets
    print("📀 Loading Disk-Based Dataset...", flush=True)
    full_dataset = DiskFeatureDataset(CACHE_DIR, max_samples=MAX_SAMPLES)
    
    # Split into Train/Val (95/5)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📈 Dataset ready. Train: {train_size}, Val: {val_size}", flush=True)
    
    # 3. Calculate dynamic class weights (use cached labels)
    print("📊 Calculating dynamic class weights...", flush=True)
    all_labels = torch.cat([m['labels'] for m in full_dataset.chunk_metadata], dim=0)
    train_indices = dataset_train.indices
    train_labels = all_labels[train_indices]
    
    pos_counts = train_labels.sum(dim=0)
    total_samples = train_labels.size(0)
    pos_weights = (total_samples / (pos_counts + 1)).to(device)
    pos_weights = torch.clamp(pos_weights, 1.0, 15.0) 
    print(f"   Weights: {pos_weights.cpu().numpy()}", flush=True)

    # 4. Training Loop
    print("⚡ Starting high-speed ClinicalHead adaptation...", flush=True)
    
    optimizer = optim.AdamW(clinical_head.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    
    dl_train = get_chunk_optimized_loader(dataset_train, batch_size=512, shuffle=True)
    dl_val = get_chunk_optimized_loader(dataset_val, batch_size=512, shuffle=False)
    
    best_macro_f1 = 0.0
    EPOCHS = 20
    best_state_dict = None
    best_thresholds_final = None
    history = []
    
    # Initialize CSV header
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "macro_f1_fixed", "macro_f1_opt"])
    
    for epoch in range(EPOCHS):
        clinical_head.train()
        total_loss = 0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = clinical_head(x)
            loss = asymmetric_focal_loss(logits, y, pos_weights)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        clinical_head.eval()
        val_probs = []
        val_targets_list = []
        with torch.no_grad():
            for x, y in tqdm(dl_val, desc=f"Epoch {epoch+1}/{EPOCHS} [Eval]"):
                x = x.to(device)
                logits = clinical_head(x)
                probs = torch.sigmoid(logits).cpu()
                val_probs.append(probs)
                val_targets_list.append(y)
                
        all_probs = torch.cat(val_probs, dim=0).numpy()
        all_targets = torch.cat(val_targets_list, dim=0).numpy()
        all_preds_fixed = (all_probs >= 0.5).astype(float)
        macro_f1_fixed = f1_score(all_targets, all_preds_fixed, average='macro', zero_division=0)
        
        # Optimized Thresholding
        best_thresholds = optimize_thresholds(all_targets, all_probs)
        all_preds_opt = (all_probs >= best_thresholds).astype(float)
        macro_f1_opt = f1_score(all_targets, all_preds_opt, average='macro', zero_division=0)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(dl_train):.4f}", flush=True)
            print(f"   Fix-Thresh F1: {macro_f1_fixed:.4f} | Opt-Thresh F1: {macro_f1_opt:.4f}", flush=True)
        
        # Log to CSV
        epoch_results = [epoch + 1, total_loss / len(dl_train), macro_f1_fixed, macro_f1_opt]
        history.append(epoch_results)
        with open(HISTORY_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(epoch_results)
            
        scheduler.step(macro_f1_opt)
            
        if macro_f1_opt > best_macro_f1:
            best_macro_f1 = macro_f1_opt
            best_state_dict = {k: v.cpu() for k, v in clinical_head.state_dict().items()}
            best_thresholds_final = best_thresholds

    # 5. Save results
    if best_state_dict is not None:
        print(f"\n🎉 Adaptation complete! Best Val Macro F1: {best_macro_f1:.4f}", flush=True)
        
        # Save summary
        summary = {
            "best_macro_f1": float(best_macro_f1),
            "best_thresholds": best_thresholds_final.tolist(),
            "epochs_total": EPOCHS,
            "final_loss": float(history[-1][1])
        }
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=4)
        
        print("Writing new weights to checkpoint...", flush=True)
        
        checkpoint['clinical_head'] = best_state_dict
        torch.save(checkpoint, OUTPUT_CKPT)
        
        with open(THRESH_PATH, 'w') as f:
            json.dump(best_thresholds_final.tolist(), f)
            
        print(f"✅ components.pt updated and thresholds saved to {THRESH_PATH}", flush=True)
    else:
        print("❌ Training failed to produce best state.", flush=True)

if __name__ == "__main__":
    main()
