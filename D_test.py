import os
import glob
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import VGGSwinHybridNet
from PIL import Image
from sklearn.metrics import (classification_report, multilabel_confusion_matrix,
                             confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, cohen_kappa_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import calibration_curve
import config

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR      = config.ROI_IMAGE_DIR
METADATA_PATH = config.METADATA_PATH
MODEL_DIR     = config.CHECKPOINT_DIR
RESULTS_DIR   = config.RESULTS_DIR
IMG_SIZE      = config.IMG_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOTSTRAP = 1000

os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_CLASSES = config.TARGET_CLASSES
NUM_CLASSES = config.NUM_CLASSES
print(f"-> {NUM_CLASSES}-Class Dataset: {TARGET_CLASSES}")


# ─── DATASET ─────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['Image_ID'])).convert('RGB')
        labels = torch.tensor(row[TARGET_CLASSES].values.astype(float), dtype=torch.float32)
        if self.transform: img = self.transform(img)
        return img, labels


# ─── PLOTS ───────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    if config.SINGLE_LABEL_MODE:
        y_true_idx = y_true.argmax(axis=1)
        y_pred_idx = y_pred.argmax(axis=1)
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_CLASSES)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.title('Unified Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png')
        plt.close()
    else:
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        cols = 2 if NUM_CLASSES in (2, 4) else min(3, NUM_CLASSES)
        rows = (NUM_CLASSES + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.ravel()
        for i, (m, name) in enumerate(zip(mcm, TARGET_CLASSES)):
            sns.heatmap(m, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'Confusion Matrix: {name}')
            axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/confusion_matrices.png')
        plt.close()

def calculate_auc_ci(y_true, y_probs, n=N_BOOTSTRAP):
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        scores.append(roc_auc_score(y_true[idx], y_probs[idx]))
    scores = np.sort(scores)
    return scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores))]

def plot_roc_curves(y_true, y_probs):
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(TARGET_CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.savefig(f'{RESULTS_DIR}/roc_curves.png'); plt.close()

def plot_pr_curves(y_true, y_probs):
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(TARGET_CLASSES):
        prec, rec, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        ap = average_precision_score(y_true[:, i], y_probs[:, i])
        plt.plot(rec, prec, label=f'{name} (AP={ap:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curves')
    plt.legend(loc='lower left'); plt.grid(alpha=0.3)
    plt.savefig(f'{RESULTS_DIR}/pr_curves.png'); plt.close()

def plot_calibration_curves(y_true, y_probs):
    cols = 2 if NUM_CLASSES in (2, 4) else min(3, NUM_CLASSES)
    rows = (NUM_CLASSES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.ravel()
    for i, name in enumerate(TARGET_CLASSES):
        pt, pp = calibration_curve(y_true[:, i], y_probs[:, i], n_bins=10, strategy='uniform')
        axes[i].plot(pp, pt, marker='o', label=name)
        axes[i].plot([0,1],[0,1],'--', label='Perfect')
        axes[i].set_title(f'Calibration: {name}')
        axes[i].legend()
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/calibration_curves.png'); plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    full_df = pd.read_csv(METADATA_PATH)

    if config.NUM_CLASSES == 2:
        print("\nFiltering dataset for pure Binary Classification (Normal vs Effusion)...")
        all_diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal']
        if 'Pneumonia' in full_df.columns:
            all_diseases.append('Pneumonia')
        full_df['disease_sum'] = full_df[all_diseases].sum(axis=1)
        mask = (full_df['disease_sum'] == 1) & ((full_df['Normal'] == 1) | (full_df['Effusion'] == 1))
        full_df = full_df[mask].copy()
        full_df.drop(columns=['disease_sum'], inplace=True)
        print(f"Filtered dataset size: {len(full_df)} images.")

    unique_df = full_df.drop_duplicates(subset='Image_ID').copy()

    # Reproduce the EXACT same Fold 0 test split as C_train.py
    test_ids_path = f"{RESULTS_DIR}/test_ids_fold0.csv"
    if os.path.exists(test_ids_path):
        test_ids = set(pd.read_csv(test_ids_path)['Image_ID'].tolist())
        test_df = unique_df[unique_df['Image_ID'].isin(test_ids)].copy()
        print(f"Loaded test split from {test_ids_path}: {len(test_df)} images")
    else:
        # Fallback: reproduce split with same seed
        from sklearn.model_selection import GroupShuffleSplit
        def get_patient_id(img_id):
            if img_id.startswith('nih_'): return img_id.split('_')[1]
            for p in img_id.split('_'):
                if 'patient' in p: return p
            return img_id
        unique_df['Patient_Group'] = unique_df['Image_ID'].apply(get_patient_id)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        _, temp_idx = next(gss1.split(unique_df, groups=unique_df['Patient_Group']))
        temp_df = unique_df.iloc[temp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        _, test_idx = next(gss2.split(temp_df, groups=temp_df['Patient_Group']))
        test_df = temp_df.iloc[test_idx].copy()
        print(f"Reconstructed test split: {len(test_df)} images")

    # TTA transforms
    tta_transforms = [
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomAffine(degrees=[10,10]),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomAffine(degrees=[-10,-10]),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize(IMG_SIZE + 32),
                             transforms.CenterCrop(IMG_SIZE),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    ]

    fold_paths  = sorted(glob.glob(f"{MODEL_DIR}/best_model_fold*.pth"))
    single_path = f"{MODEL_DIR}/best_model.pth"
    model_paths = fold_paths if fold_paths else ([single_path] if os.path.exists(single_path) else [])
    if not model_paths:
        raise FileNotFoundError(f"No model weights found in {MODEL_DIR}/")
    print(f"Found {len(model_paths)} model(s) — {'ensemble' if len(model_paths)>1 else 'single'} evaluation.")

    thresh_path = f'{RESULTS_DIR}/optimal_thresholds.json'
    optimal_thresholds = ({cls: 0.5 for cls in TARGET_CLASSES}
                          if not os.path.exists(thresh_path)
                          else json.load(open(thresh_path)))

    # Collect labels once
    base_loader = DataLoader(ChestXRayDataset(test_df, DATA_DIR, tta_transforms[0]),
                             batch_size=16, num_workers=4)
    all_labels = []
    with torch.no_grad():
        for _, labels in base_loader:
            all_labels.append(labels)
    all_labels = torch.cat(all_labels).numpy()

    # TTA + ensemble
    all_probs_ensemble = []
    for m_idx, model_path in enumerate(model_paths):
        model = VGGSwinHybridNet(
            num_classes=NUM_CLASSES,
            drop_path_rate=config.DROP_PATH_RATE,
            head_dropout=config.HEAD_DROPOUT
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"  Model {m_idx+1}/{len(model_paths)}: {os.path.basename(model_path)}")

        tta_probs_list = []
        for t_idx, tta_t in enumerate(tta_transforms):
            loader = DataLoader(ChestXRayDataset(test_df, DATA_DIR, tta_t),
                                batch_size=16, num_workers=4)
            probs = []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for images, _ in loader:
                    outputs = model(images.to(DEVICE))
                    if config.SINGLE_LABEL_MODE:
                        probs.append(torch.softmax(outputs, dim=1).cpu())
                    else:
                        probs.append(torch.sigmoid(outputs).cpu())
            tta_probs_list.append(torch.cat(probs).numpy())
            print(f"    TTA {t_idx+1}/{len(tta_transforms)}")
        all_probs_ensemble.append(np.mean(tta_probs_list, axis=0))

    all_probs = np.mean(all_probs_ensemble, axis=0)

    if config.SINGLE_LABEL_MODE:
        all_preds = np.zeros_like(all_probs)
        all_preds[np.arange(len(all_probs)), all_probs.argmax(axis=1)] = 1.0
    else:
        all_preds = np.zeros_like(all_probs)
        for i, cls in enumerate(TARGET_CLASSES):
            all_preds[:, i] = (all_probs[:, i] >= optimal_thresholds[cls]).astype(float)

    # Save raw predictions for threshold_optimizer.py
    raw_df = pd.DataFrame()
    for i, cls in enumerate(TARGET_CLASSES):
        raw_df[f'true_{cls}'] = all_labels[:, i]
        raw_df[f'prob_{cls}'] = all_probs[:, i]
    raw_df.to_csv(f'{RESULTS_DIR}/test_raw_predictions.csv', index=False)

    print("\n" + "="*50)
    print("FINAL TEST SET PERFORMANCE")
    print("="*50)
    
    report_dict = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f'{RESULTS_DIR}/test_classification_report.csv')
    print(classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, zero_division=0))

    if config.SINGLE_LABEL_MODE:
        all_labels_idx = all_labels.argmax(axis=1)
        all_preds_idx = all_preds.argmax(axis=1)
        mcc = matthews_corrcoef(all_labels_idx, all_preds_idx)
        kappa = cohen_kappa_score(all_labels_idx, all_preds_idx)
        print(f"Matthew's Correlation Coefficient (MCC): {mcc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        # Add to the results list that will be saved to CSV
        extra_results = [{'Class': 'MCC', 'AUC': mcc, 'CI_Lower': mcc, 'CI_Upper': mcc, 'AP': mcc},
                         {'Class': 'Kappa', 'AUC': kappa, 'CI_Lower': kappa, 'CI_Upper': kappa, 'AP': kappa}]
    else:
        extra_results = []

    macro_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    print(f"Macro AUC: {macro_auc:.4f}")
    results = []
    for i, name in enumerate(TARGET_CLASSES):
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        lo, hi = calculate_auc_ci(all_labels[:, i], all_probs[:, i])
        ap = average_precision_score(all_labels[:, i], all_probs[:, i])
        print(f"  {name:15s}: AUC={auc:.4f} [95% CI: {lo:.4f}-{hi:.4f}] AP={ap:.4f}")
        results.append({'Class': name, 'AUC': auc, 'CI_Lower': lo, 'CI_Upper': hi, 'AP': ap})

    pd.DataFrame(results + extra_results).to_csv(f'{RESULTS_DIR}/test_performance_metrics.csv', index=False)

    print("\nGenerating plots...")
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curves(all_labels, all_probs)
    plot_pr_curves(all_labels, all_probs)
    plot_calibration_curves(all_labels, all_probs)
    print(f"Done. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
