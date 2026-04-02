import torch
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

CKPT_PATH = 'models/best/components.pt'

def check_checkpoint():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found at {CKPT_PATH}")
        return

    print(f"✅ Found checkpoint at {CKPT_PATH}")
    try:
        # Using weights_only=False because it might contain custom classes or structures
        checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        print("Keys in checkpoint:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                 print(f"  - {key}: {len(checkpoint[key])} parameters")
            else:
                 print(f"  - {key}: {type(checkpoint[key])}")
                 
        # Verify architecture match for ClinicalHead
        from aegisrad.models import ClinicalHead
        ch = ClinicalHead()
        try:
            ch.load_state_dict(checkpoint['clinical_head'])
            print("✅ ClinicalHead state_dict matches architecture.")
        except Exception as e:
            print(f"❌ ClinicalHead architecture mismatch: {e}")
            
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")

if __name__ == "__main__":
    check_checkpoint()
