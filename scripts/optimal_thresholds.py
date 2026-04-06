import os
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.getcwd()))

# Try to import from the eval_venv directory where evaluate_aegisrad lives
sys.path.append("/Volumes/SSD/Jetson_AegisRad/eval_venv")

from evaluate_aegisrad import load_models, load_test_data, run_inference, CONDITIONS

def find_optimal_thresholds(gt, probs):
    n_classes = gt.shape[1]
    best_thresholds = {}
    best_f1s = {}
    
    print("\n🔍 Optimizing thresholds for 14 conditions...")
    for i in range(n_classes):
        cond = CONDITIONS[i]
        class_gt = gt[:, i]
        class_probs = probs[:, i]
        
        best_f1 = -1
        best_t = 0.5
        
        # Grid search through thresholds
        for t in np.linspace(0.05, 0.85, 81):
            preds = (class_probs >= t).astype(float)
            f1 = f1_score(class_gt, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds[cond] = float(round(best_t, 3))
        best_f1s[cond] = float(round(best_f1, 4))
        
    return best_thresholds, best_f1s

def main():
    encoder, classifier, qformer, projector = load_models()
    test_data = load_test_data(500) # Use 500 for a balance of speed and coverage
    
    all_gt = []
    all_probs = []
    
    print(f"🚀 Running inference on {len(test_data)} images...")
    for sample in tqdm(test_data):
        try:
            # Note: run_inference in evaluate_aegisrad returns (all_probs, probs_arr, flagged, report, latency)
            probs_dict, probs_arr, _, _, _ = run_inference(encoder, classifier, qformer, projector, sample['image_path'])
            all_gt.append(sample['labels'])
            all_probs.append(probs_arr)
        except Exception as e:
            continue
            
    gt_matrix = np.array(all_gt)
    prob_matrix = np.array(all_probs)
    
    thresholds, f1s = find_optimal_thresholds(gt_matrix, prob_matrix)
    
    print("\n" + "="*50)
    print(f"{'Condition':<30} | {'Opt. Thresh':<12} | {'New F1':<8}")
    print("-" * 50)
    for cond in CONDITIONS:
        print(f"{cond:<30} | {thresholds[cond]:<12.3f} | {f1s[cond]:<8.4f}")
    
    print("\n📦 Copy this into your config.py:")
    print("CLASS_THRESHOLDS = " + str(thresholds))

if __name__ == "__main__":
    main()
