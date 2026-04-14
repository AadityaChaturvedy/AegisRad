import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
import json

def compute_pathology_metrics(y_true, y_probs, conditions):
    """
    Computes per-class F1, AUC, and multi-class stats.
    y_true: [N, 14] binary labels
    y_probs: [N, 14] probabilities
    """
    y_pred = (y_probs > 0.5).astype(int)
    
    metrics = {}
    
    # 1. Macro Averaged Stats
    metrics['overall'] = {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro')),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro')),
        'mean_auc': float(roc_auc_score(y_true, y_probs, average='macro'))
    }
    
    # 2. Per-Class Stats
    per_class = {}
    for i, condition in enumerate(conditions):
        try:
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except:
            auc = 0.5 # Default for no variance
            
        cm = confusion_matrix(y_true[:, i], y_pred[:, i]).tolist()
        
        per_class[condition] = {
            'f1': float(f1_score(y_true[:, i], y_pred[:, i])),
            'auc': float(auc),
            'confusion_matrix': cm
        }
        
    metrics['per_class'] = per_class
    return metrics

def compute_language_metrics(references, candidates):
    """Computes ROUGE-L similarity between ground truth and generated reports."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    scores = []
    for ref, cand in zip(references, candidates):
        # Flatten strings
        s = scorer.score(ref, cand)
        scores.append(s['rougeL'].fmeasure)
        
    return {
        'mean_rouge_l': float(np.mean(scores)),
        'std_rouge_l': float(np.std(scores))
    }

def save_benchmark(metrics, resource_usage, output_path="bench_results.json"):
    """Dumps all data to a JSON for the paper draft."""
    data = {
        "metrics": metrics,
        "resources": resource_usage,
        "metadata": {
            "platform": "NVIDIA Jetson Nano 4GB",
            "benchmark_date": pd.Timestamp.now().isoformat()
        }
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"📊 Benchmarks saved to {output_path}")
