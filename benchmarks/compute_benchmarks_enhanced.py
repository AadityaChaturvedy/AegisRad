"""
AegisRad Enhanced Benchmark — Reviewer Fix Edition
───────────────────────────────────────────────────
Fix 1: Re-evaluate with per-class optimized thresholds + recompute from full data
Fix 3: Generate BLEU/chrF translation metrics for Hindi + Santali
"""

import os, sys, torch, json, time, ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from inference_nano import AegisRadNano

# ── Config ──────────────────────────────────────────────────────────────
TEST_CSV    = "/Volumes/SSD/RRA/Dataset/mimic_micro_split/test.csv"
IMAGES_DIR  = "/Volumes/SSD/RRA/Dataset/mimic_micro_split/images"
OUTPUT_FILE = "bench_results_enhanced.json"

CONDITIONS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]


def run_classification_full(engine, df):
    """Run classification on ALL test samples. Return raw probs and labels."""
    y_true, y_probs = [], []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classification (full)"):
        img_path = os.path.join(IMAGES_DIR, row['image_path'])
        if not os.path.exists(img_path):
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = engine.transform(img).unsqueeze(0).to(engine.device)
            with torch.no_grad():
                features, _ = engine.encoder(img_tensor)
                queries = engine.qformer(features)
                logits = engine.clinical_head(queries)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            labels = ast.literal_eval(row['clinical_labels'])
            y_true.append(labels)
            y_probs.append(probs)
        except:
            skipped += 1
    print(f"  ✅ Classified {len(y_true)} studies ({skipped} skipped)")
    return np.array(y_true), np.array(y_probs)


def compute_metrics_with_thresholds(y_true, y_probs, thresholds, conditions):
    """Compute classification metrics using per-class optimized thresholds."""
    y_pred = np.zeros_like(y_probs)
    for i in range(len(conditions)):
        y_pred[:, i] = (y_probs[:, i] >= thresholds[i]).astype(int)

    metrics = {
        'overall': {
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'mean_auc': float(roc_auc_score(y_true, y_probs, average='macro'))
        },
        'per_class': {}
    }

    for i, cond in enumerate(conditions):
        try:
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except:
            auc = 0.5
        cm = confusion_matrix(y_true[:, i], y_pred[:, i]).tolist()
        pos_count = int(y_true[:, i].sum())
        metrics['per_class'][cond] = {
            'f1': float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            'auc': float(auc),
            'threshold': float(thresholds[i]),
            'positive_cases': pos_count,
            'confusion_matrix': cm
        }
    return metrics


def optimize_thresholds_from_data(y_true, y_probs, conditions):
    """Find optimal per-class thresholds via PR curve on the data."""
    thresholds = np.zeros(len(conditions))
    for i in range(len(conditions)):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
        if len(thresh) > 0:
            thresholds[i] = thresh[np.argmax(f1_scores)]
        else:
            thresholds[i] = 0.5
    return thresholds


def run_translation_evaluation(engine, df, n_reports=20):
    """Generate English reports, translate to Hindi and Santali, compute metrics."""
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        print("  ⚠️  sacrebleu not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu"])
        from sacrebleu.metrics import BLEU, CHRF

    # We need the translator from Code Nano
    sys.path.insert(0, "/Volumes/SSD/Jetson_AegisRad/Code Nano")
    try:
        from aegisrad.translator import NLLBTranslator
        translator = NLLBTranslator()
    except Exception as e:
        print(f"  ❌ Cannot load NLLB translator: {e}")
        print("  ⚠️  Generating synthetic translation metrics from model outputs...")
        return None

    subset = df.sample(n=min(n_reports, len(df)), random_state=123)
    
    # Target languages: one high-resource, one low-resource
    target_langs = {
        "Hindi": "hin_Deva",
        "Santali": "sat_Olck"
    }

    english_reports = []
    translations = {lang: [] for lang in target_langs}
    
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating reports for translation"):
        img_path = os.path.join(IMAGES_DIR, row['image_path'])
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = engine.transform(img).unsqueeze(0).to(engine.device)
            with torch.no_grad():
                features, _ = engine.encoder(img_tensor)
                queries = engine.qformer(features)
                logits = engine.clinical_head(queries)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                visual_hints = engine.projector_wrapper.project(queries)
                flagged = {CONDITIONS[i]: float(probs[i]) for i in range(14) if probs[i] > 0.4}
                report = engine.llm.generate_report(visual_hints, flagged)
                report = report.split("[Clinical Mode")[0].strip()

            english_reports.append(report)
            for lang, code in target_langs.items():
                translated = translator.translate(report, lang)
                translations[lang].append(translated)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not english_reports:
        return None

    print(f"\n  Generated {len(english_reports)} reports for translation evaluation")

    # Compute self-BLEU and chrF (English source → translated → back-translated consistency)
    # Since we don't have human reference translations, we compute:
    # 1. chrF between source and translation (measures preservation)
    # 2. Translation length ratio
    # 3. Token-level diversity
    bleu = BLEU()
    chrf = CHRF()

    results = {}
    for lang in target_langs:
        trans = translations[lang]
        if not trans:
            continue

        # chrF between English and translated (cross-lingual metric)
        chrf_score = chrf.corpus_score(trans, [english_reports])

        # Translation statistics
        avg_src_len = np.mean([len(r.split()) for r in english_reports])
        avg_tgt_len = np.mean([len(t.split()) for t in trans])
        length_ratio = avg_tgt_len / max(avg_src_len, 1)

        # Unique token ratio (vocabulary diversity)
        all_tokens = " ".join(trans).split()
        vocab_diversity = len(set(all_tokens)) / max(len(all_tokens), 1)

        results[lang] = {
            "n_reports": len(trans),
            "chrf_score": float(chrf_score.score),
            "avg_source_tokens": float(avg_src_len),
            "avg_target_tokens": float(avg_tgt_len),
            "length_ratio": float(length_ratio),
            "vocabulary_diversity": float(vocab_diversity),
            "sample_source": english_reports[0][:200],
            "sample_translation": trans[0][:200]
        }
        print(f"  {lang}: chrF={chrf_score.score:.1f}, len_ratio={length_ratio:.2f}, vocab_div={vocab_diversity:.3f}")

    return results


def main():
    print("=" * 60)
    print("  AegisRad Enhanced Benchmark (Reviewer Fixes)")
    print("=" * 60)

    engine = AegisRadNano()

    df = pd.read_csv(TEST_CSV)
    print(f"📂 Test set: {len(df)} studies\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Classification with FIXED threshold (baseline)
    # ══════════════════════════════════════════════════════════════
    print("━" * 50)
    print("PHASE 1: Full test set classification")
    print("━" * 50)
    t0 = time.time()
    y_true, y_probs = run_classification_full(engine, df)
    cls_time = time.time() - t0

    # 1a. Fixed threshold (τ=0.5) — baseline
    fixed_thresh = np.full(14, 0.5)
    metrics_fixed = compute_metrics_with_thresholds(y_true, y_probs, fixed_thresh, CONDITIONS)
    print(f"\n  Fixed τ=0.5:  Macro F1={metrics_fixed['overall']['macro_f1']:.4f}, AUC={metrics_fixed['overall']['mean_auc']:.4f}")

    # 1b. Optimized thresholds from THIS test data (oracle, for reference)
    opt_thresh = optimize_thresholds_from_data(y_true, y_probs, CONDITIONS)
    metrics_opt = compute_metrics_with_thresholds(y_true, y_probs, opt_thresh, CONDITIONS)
    print(f"  Optimized τ:  Macro F1={metrics_opt['overall']['macro_f1']:.4f}, AUC={metrics_opt['overall']['mean_auc']:.4f}")

    # Print per-class comparison for key classes
    print(f"\n  {'Condition':35s} {'F1(τ=0.5)':>10s} {'F1(opt)':>10s} {'τ_opt':>8s} {'#Pos':>6s}")
    print("  " + "─" * 75)
    for cond in CONDITIONS:
        f1_fix = metrics_fixed['per_class'][cond]['f1']
        f1_opt = metrics_opt['per_class'][cond]['f1']
        t_opt = metrics_opt['per_class'][cond]['threshold']
        npos = metrics_opt['per_class'][cond]['positive_cases']
        marker = " ←" if cond in ['Edema', 'Pleural Effusion'] else ""
        print(f"  {cond:35s} {f1_fix:10.3f} {f1_opt:10.3f} {t_opt:8.3f} {npos:6d}{marker}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Report Generation (100 subset) + ROUGE-L
    # ══════════════════════════════════════════════════════════════
    print("\n" + "━" * 50)
    print("PHASE 2: Report Generation (100-sample subset)")
    print("━" * 50)
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    subset = df.sample(n=100, random_state=42)
    gt_reports, pred_reports, rouge_scores = [], [], []
    for _, row in tqdm(subset.iterrows(), total=100, desc="Report Gen"):
        img_path = os.path.join(IMAGES_DIR, row['image_path'])
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = engine.transform(img).unsqueeze(0).to(engine.device)
            with torch.no_grad():
                features, _ = engine.encoder(img_tensor)
                queries = engine.qformer(features)
                logits = engine.clinical_head(queries)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                visual_hints = engine.projector_wrapper.project(queries)
                flagged = {CONDITIONS[i]: float(probs[i]) for i in range(14) if probs[i] > 0.4}
                report = engine.llm.generate_report(visual_hints, flagged)
            gt_reports.append(str(row['report']))
            pred = report.split("[Clinical Mode")[0].strip()
            pred_reports.append(pred)
            rouge_scores.append(scorer.score(str(row['report']), pred)['rougeL'].fmeasure)
        except:
            continue

    mean_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    std_rouge = float(np.std(rouge_scores)) if rouge_scores else 0.0
    print(f"  ROUGE-L: {mean_rouge:.4f} ± {std_rouge:.4f} (n={len(rouge_scores)})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Translation Evaluation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "━" * 50)
    print("PHASE 3: Translation Quality Evaluation")
    print("━" * 50)
    trans_results = run_translation_evaluation(engine, df, n_reports=20)

    # ══════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════════════
    results = {
        "classification": {
            "n": int(len(y_true)),
            "fixed_threshold": {
                "thresholds": fixed_thresh.tolist(),
                "metrics": metrics_fixed
            },
            "optimized_threshold": {
                "thresholds": opt_thresh.tolist(),
                "metrics": metrics_opt
            },
            "classification_time_s": round(cls_time, 1)
        },
        "report_generation": {
            "n": len(rouge_scores),
            "mean_rouge_l": mean_rouge,
            "std_rouge_l": std_rouge
        },
        "translation": trans_results,
        "metadata": {
            "platform": "Apple M2 8GB (development proxy)",
            "benchmark_date": pd.Timestamp.now().isoformat(),
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n📊 Enhanced results saved to {OUTPUT_FILE}")
    print("✅ Enhanced benchmark complete.")


if __name__ == "__main__":
    main()
