import os
import sys

# Setup relative paths to load config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score,
                             multilabel_confusion_matrix, matthews_corrcoef, cohen_kappa_score)
from sklearn.model_selection import GroupShuffleSplit
import config

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(parent_dir, config.ROI_IMAGE_DIR)
METADATA_PATH = os.path.join(parent_dir, config.METADATA_PATH)
RESULTS_DIR   = os.path.join(parent_dir, getattr(config, 'RESULTS_DIR', 'results'))
CHECKPOINT_DIR = os.path.join(parent_dir, getattr(config, 'CHECKPOINT_DIR', 'weights'))
LOG_FILE      = os.path.join(RESULTS_DIR, "baseline_vgg16_log.csv")

BATCH_SIZE         = config.BATCH_SIZE
ACCUMULATION_STEPS = config.ACCUMULATION_STEPS
LABEL_SMOOTHING    = config.LABEL_SMOOTHING
MIXUP_ALPHA        = getattr(config, 'MIXUP_ALPHA', 0.2)
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
MAX_GRAD_NORM      = getattr(config, 'MAX_GRAD_NORM', 5.0)
IMG_SIZE           = config.IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable TensorFloat-32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

TARGET_CLASSES = config.TARGET_CLASSES
NUM_CLASSES = config.NUM_CLASSES
print(f"-> Using {NUM_CLASSES}-Class Dataset: {TARGET_CLASSES}")


# ─── DATASET ─────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_ID'])
        image = Image.open(img_path).convert('RGB')
        labels = torch.tensor(row[TARGET_CLASSES].values.astype(float), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels


def mixup_batch(images, labels, alpha=MIXUP_ALPHA):
    if alpha <= 0: return images, labels
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx], lam * labels + (1 - lam) * labels[idx]


# ─── EVALUATION ──────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    val_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            if getattr(config, 'SINGLE_LABEL_MODE', True):
                all_probs.append(torch.softmax(outputs, dim=1).cpu())
            else:
                all_probs.append(torch.sigmoid(outputs).cpu())
            all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if getattr(config, 'SINGLE_LABEL_MODE', True):
        all_preds = np.zeros_like(all_probs)
        all_preds[np.arange(len(all_probs)), all_probs.argmax(axis=1)] = 1.0
        all_labels_idx = all_labels.argmax(axis=1)
        all_preds_idx = all_preds.argmax(axis=1)
        accuracy = accuracy_score(all_labels_idx, all_preds_idx)
        mcc = matthews_corrcoef(all_labels_idx, all_preds_idx)
        kappa = cohen_kappa_score(all_labels_idx, all_preds_idx)
    else:
        all_preds  = (all_probs > 0.5).astype(float)
        accuracy = accuracy_score(all_labels, all_preds)
        mcc = 0.0
        kappa = 0.0

    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    specificities = []
    for m in mcm:
        tn, fp, fn, tp = m.ravel()
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    try:
        auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        auc_macro = float('nan')

    metrics = {
        'loss': val_loss / len(loader),
        'accuracy':      accuracy,
        'f1_macro':      f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro':  recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'specificity_macro': np.mean(specificities),
        'auc_macro':     auc_macro,
        'pr_auc_macro':  average_precision_score(all_labels, all_probs, average='macro'),
        'mcc':           mcc,
        'kappa':         kappa
    }
    for i, cls in enumerate(TARGET_CLASSES):
        metrics[f'f1_{cls}'] = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        try:
            metrics[f'auc_{cls}'] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            metrics[f'auc_{cls}'] = float('nan')
        metrics[f'pr_auc_{cls}'] = average_precision_score(all_labels[:, i], all_probs[:, i])
        metrics[f'spec_{cls}']   = specificities[i]
    return metrics


def log_to_csv(data_dict):
    df = pd.DataFrame([data_dict])
    if not os.path.isfile(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()
    FOLD = args.fold
    FOLD_SEED = 42 + FOLD * 100

    MODEL_SAVE_PATH = f"{CHECKPOINT_DIR}/best_vgg16_fold{FOLD}.pth"
    checkpoint_path = f"{CHECKPOINT_DIR}/latest_vgg16_fold{FOLD}.pth"

    print(f"\n{'='*50}")
    print(f"  VGG16 BASELINE | FOLD {FOLD} | {IMG_SIZE}×{IMG_SIZE} | {NUM_CLASSES} classes")
    print(f"  Batch: {BATCH_SIZE} × {ACCUMULATION_STEPS} steps = {BATCH_SIZE*ACCUMULATION_STEPS} effective")
    print(f"{'='*50}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_df = pd.read_csv(METADATA_PATH)

    if getattr(config, 'SINGLE_LABEL_MODE', True):
        print("\n--- Applying SINGLE-LABEL MULTICLASS Filtering ---")
        full_df['Total_Labels'] = full_df[TARGET_CLASSES].sum(axis=1)
        full_df = full_df[full_df['Total_Labels'] == 1].copy()
        full_df.drop(columns=['Total_Labels'], inplace=True)
        print(f"Filtered dataset size: {len(full_df)} exclusive single-label images.")

    def get_patient_id(img_id):
        if img_id.startswith('nih_'): return img_id.split('_')[1]
        if 'patient' in img_id:
            for p in img_id.split('_'):
                if 'patient' in p: return p
        return img_id

    full_df['Patient_Group'] = full_df['Image_ID'].apply(get_patient_id)
    unique_df = full_df.drop_duplicates(subset='Image_ID').copy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=FOLD_SEED)
    train_idx, temp_idx = next(gss1.split(unique_df, groups=unique_df['Patient_Group']))
    train_ids = set(unique_df.iloc[train_idx]['Image_ID'])
    temp_ids  = set(unique_df.iloc[temp_idx]['Image_ID'])

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=FOLD_SEED)
    temp_unique = unique_df[unique_df['Image_ID'].isin(temp_ids)]
    val_idx, test_idx = next(gss2.split(temp_unique, groups=temp_unique['Patient_Group']))
    val_ids  = set(temp_unique.iloc[val_idx]['Image_ID'])
    test_ids = set(temp_unique.iloc[test_idx]['Image_ID'])

    val_df   = unique_df[unique_df['Image_ID'].isin(val_ids)].copy()
    test_df  = unique_df[unique_df['Image_ID'].isin(test_ids)].copy()

    if getattr(config, 'SINGLE_LABEL_MODE', True):
        base_train_df = unique_df[unique_df['Image_ID'].isin(train_ids)].copy()
        print(f"\nBalancing Training Set to exactly {config.SAMPLES_PER_CLASS} samples per class...")
        sampled_dfs = []
        for cls in TARGET_CLASSES:
            cls_df = base_train_df[base_train_df[cls] == 1]
            if len(cls_df) >= config.SAMPLES_PER_CLASS:
                sampled_dfs.append(cls_df.sample(n=config.SAMPLES_PER_CLASS, replace=False, random_state=42))
            elif len(cls_df) > 0:
                sampled_dfs.append(cls_df.sample(n=config.SAMPLES_PER_CLASS, replace=True, random_state=42))
        train_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        train_df = full_df[full_df['Image_ID'].isin(train_ids)].copy()

    print(f"Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    NUM_WORKERS = min(os.cpu_count() or 4, 12)
    train_loader = DataLoader(ChestXRayDataset(train_df, DATA_DIR, train_trans),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(ChestXRayDataset(val_df, DATA_DIR, val_trans),
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ─── VGG16 BASELINE MODEL ───
    model = models.vgg16_bn(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    model = model.to(DEVICE)

    def criterion(inputs, targets):
        smooth = targets * (1 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / NUM_CLASSES)
        return torch.nn.functional.cross_entropy(inputs, smooth)

    # Fast Baseline Phase schedule: (name, epochs, lr, freeze_backbone)
    phases = [
        ("Warmup",  2,  1e-3, True),
        ("Phase_1", 8, config.LR, False),
    ]

    best_val_auc = 0.0
    best_val_f1  = 0.0
    start_phase_idx = 0
    start_epoch = 1

    for phase_idx in range(start_phase_idx, len(phases)):
        phase_name, epochs, lr, freeze_backbone = phases[phase_idx]
        print(f"\n>>> {phase_name} (LR={lr})")

        for p in model.features.parameters(): 
            p.requires_grad = not freeze_backbone

        optimizer_class = getattr(torch.optim, getattr(config, "OPTIMIZER", "AdamW"))
        optimizer = optimizer_class(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            # Freeze BatchNorm stats for backbone
            model.features.eval()

            train_loss = 0
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"[{phase_name}] E{epoch}/{epochs}")
            
            for step, (images, labels) in enumerate(pbar):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images, labels = mixup_batch(images, labels)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    loss = criterion(model(images), labels) / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * ACCUMULATION_STEPS
                pbar.set_postfix({'loss': f"{train_loss/(step+1):.4f}"})

            val_metrics = evaluate(model, val_loader, criterion)
            scheduler.step()
            epoch_time = time.time() - t0

            log_to_csv({'Phase': phase_name, 'Epoch': epoch,
                        'LR': optimizer.param_groups[0]['lr'],
                        'Time_Sec': epoch_time,
                        'Train_Loss': train_loss / len(train_loader),
                        **{f'Val_{k}': v for k, v in val_metrics.items()}})

            print(f"  Loss:{val_metrics['loss']:.4f} | Acc:{val_metrics['accuracy']:.4f} | F1:{val_metrics['f1_macro']:.4f} "
                  f"| AUC:{val_metrics['auc_macro']:.4f} | MCC:{val_metrics['mcc']:.4f}")

            if val_metrics['auc_macro'] > best_val_auc:
                best_val_auc = val_metrics['auc_macro']
                best_val_f1  = val_metrics['f1_macro']
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  🚀 Best model saved (AUC={best_val_auc:.4f})")

    print(f"\n✅ Done. Best VGG16 Val AUC={best_val_auc:.4f} | F1={best_val_f1:.4f}")

if __name__ == "__main__":
    main()
