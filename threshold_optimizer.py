import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import config

def find_optimal_thresholds(probs, labels, target_classes):
    """Find optimal threshold per class using Youden's J = Sensitivity + Specificity - 1."""
    thresholds = {}
    print("=" * 60)
    print("  PER-CLASS OPTIMAL THRESHOLD ANALYSIS (384x384)")
    print("=" * 60)

    for i, cls in enumerate(target_classes):
        fpr, tpr, thresh_vals = roc_curve(labels[:, i], probs[:, i])
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_thresh = float(thresh_vals[best_idx])

        default_preds = (probs[:, i] > 0.5).astype(float)
        optimal_preds = (probs[:, i] > best_thresh).astype(float)

        f1_default = f1_score(labels[:, i], default_preds, zero_division=0)
        f1_optimal = f1_score(labels[:, i], optimal_preds, zero_division=0)

        thresholds[cls] = best_thresh
        direction = "+" if f1_optimal >= f1_default else ""
        print(f"\n  {cls}:")
        print(f"    Threshold 0.500  →  F1: {f1_default:.4f}")
        print(f"    Threshold {best_thresh:.3f}  →  F1: {f1_optimal:.4f}  ({direction}{f1_optimal - f1_default:.4f})")

    return thresholds


def apply_thresholds(probs, labels, thresholds, target_classes):
    """Apply per-class thresholds and compute final metrics."""
    all_preds_default = (probs > 0.5).astype(float)
    all_preds_optimal = np.zeros_like(probs)
    for i, cls in enumerate(target_classes):
        all_preds_optimal[:, i] = (probs[:, i] > thresholds[cls]).astype(float)

    f1_default = f1_score(labels, all_preds_default, average='macro', zero_division=0)
    f1_optimal = f1_score(labels, all_preds_optimal, average='macro', zero_division=0)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Macro F1 (default threshold=0.5):  {f1_default:.4f}")
    print(f"  Macro F1 (optimal thresholds):     {f1_optimal:.4f}  (+{f1_optimal - f1_default:.4f})")
    print("\n  Per-class optimal thresholds:")
    for cls, t in thresholds.items():
        print(f"    {cls}: {t:.4f}")

    return all_preds_optimal


def main():
    csv_path = f'{config.RESULTS_DIR}/test_raw_predictions.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        print("Please run C_test_384.py first so it can generate the raw predictions.")
        return

    # Automatically extract target classes from columns
    target_classes = [col.replace('prob_', '') for col in df.columns if col.startswith('prob_')]
    print(f"Detected {len(target_classes)} classes from predictions: {target_classes}\n")

    probs  = np.stack([df[f'prob_{c}'].values for c in target_classes], axis=1)
    labels = np.stack([df[f'true_{c}'].values for c in target_classes], axis=1)

    thresholds = find_optimal_thresholds(probs, labels, target_classes)
    apply_thresholds(probs, labels, thresholds, target_classes)

    # Save thresholds to JSON so C_test_384.py can use them automatically next time
    out_path = f'{config.RESULTS_DIR}/optimal_thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  Optimal thresholds saved to {out_path}")
    print("  (If you run C_test_384.py again, it will automatically use these thresholds!)")


if __name__ == '__main__':
    main()
