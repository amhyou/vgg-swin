import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import VGGSwinHybridNet
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score,
                             multilabel_confusion_matrix, matthews_corrcoef, cohen_kappa_score)
from sklearn.model_selection import GroupShuffleSplit
import config

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR      = config.ROI_IMAGE_DIR
METADATA_PATH = config.METADATA_PATH
LOG_FILE      = config.LOG_FILE
CHECKPOINT_DIR = config.CHECKPOINT_DIR

BATCH_SIZE         = config.BATCH_SIZE
ACCUMULATION_STEPS = config.ACCUMULATION_STEPS
LABEL_SMOOTHING    = config.LABEL_SMOOTHING
MIXUP_ALPHA        = config.MIXUP_ALPHA
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
MAX_GRAD_NORM      = config.MAX_GRAD_NORM
IMG_SIZE           = config.IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable TensorFloat-32 — free ~2-3x speedup on Ampere/Blackwell with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms

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
            if config.SINGLE_LABEL_MODE:
                all_probs.append(torch.softmax(outputs, dim=1).cpu())
            else:
                all_probs.append(torch.sigmoid(outputs).cpu())
            all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if config.SINGLE_LABEL_MODE:
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

    MODEL_SAVE_PATH = f"{CHECKPOINT_DIR}/best_model_fold{FOLD}.pth"
    checkpoint_path = f"{CHECKPOINT_DIR}/latest_checkpoint_fold{FOLD}.pth"

    print(f"\n{'='*50}")
    print(f"  TRAINING FOLD {FOLD} | {IMG_SIZE}×{IMG_SIZE} | {NUM_CLASSES} classes")
    print(f"  Batch: {BATCH_SIZE} × {ACCUMULATION_STEPS} steps = {BATCH_SIZE*ACCUMULATION_STEPS} effective")
    print(f"{'='*50}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    full_df = pd.read_csv(METADATA_PATH)

    if config.SINGLE_LABEL_MODE:
        print("\n--- Applying SINGLE-LABEL MULTICLASS Filtering ---")
        full_df['Total_Labels'] = full_df[TARGET_CLASSES].sum(axis=1)
        full_df = full_df[full_df['Total_Labels'] == 1].copy()
        full_df.drop(columns=['Total_Labels'], inplace=True)
        print(f"Filtered dataset size: {len(full_df)} exclusive single-label images.")
    elif config.NUM_CLASSES == 2:
        print("\nFiltering dataset for pure Binary Classification (Normal vs Effusion)...")
        # Keep rows where only Effusion is 1, OR only Normal is 1, and everything else is 0
        all_diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal']
        if 'Pneumonia' in full_df.columns:
            all_diseases.append('Pneumonia')

        # Calculate sum of all disease labels
        full_df['disease_sum'] = full_df[all_diseases].sum(axis=1)

        # Condition: Exactly 1 disease is active, AND it's either Normal or Effusion
        mask = (full_df['disease_sum'] == 1) & ((full_df['Normal'] == 1) | (full_df['Effusion'] == 1))
        full_df = full_df[mask].copy()
        full_df.drop(columns=['disease_sum'], inplace=True)
        print(f"Filtered dataset size: {len(full_df)} images.")

    # Extract patient group from Image_ID to prevent data leakage
    def get_patient_id(img_id):
        if img_id.startswith('nih_'):
            return img_id.split('_')[1]
        if 'patient' in img_id:
            for p in img_id.split('_'):
                if 'patient' in p: return p
        return img_id

    full_df['Patient_Group'] = full_df['Image_ID'].apply(get_patient_id)

    # Split on UNIQUE images to avoid leakage from duplicate rows
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

    # Validation and test use UNIQUE images only (unbiased metrics)
    val_df   = unique_df[unique_df['Image_ID'].isin(val_ids)].copy()
    test_df  = unique_df[unique_df['Image_ID'].isin(test_ids)].copy()

    if config.SINGLE_LABEL_MODE:
        base_train_df = unique_df[unique_df['Image_ID'].isin(train_ids)].copy()
        print(f"\nBalancing Training Set to exactly {config.SAMPLES_PER_CLASS} samples per class...")
        sampled_dfs = []
        for cls in TARGET_CLASSES:
            cls_df = base_train_df[base_train_df[cls] == 1]
            if len(cls_df) >= config.SAMPLES_PER_CLASS:
                # Undersample unique images
                sampled_dfs.append(cls_df.sample(n=config.SAMPLES_PER_CLASS, replace=False, random_state=42))
            elif len(cls_df) > 0:
                # Oversample with replacement
                print(f"  -> Oversampling {cls}: from {len(cls_df)} up to {config.SAMPLES_PER_CLASS} images")
                sampled_dfs.append(cls_df.sample(n=config.SAMPLES_PER_CLASS, replace=True, random_state=42))
            else:
                print(f"  -> WARNING: Class {cls} has 0 training images! Skipping.")
                
        train_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Training keeps ALL duplicate rows (oversampling effect from B_preprocess.py)
        train_df = full_df[full_df['Image_ID'].isin(train_ids)].copy()

    # Save test IDs for D_test.py to reproduce the exact same split
    test_df[['Image_ID']].to_csv(f"{config.RESULTS_DIR}/test_ids_fold{FOLD}.csv", index=False)

    print(f"Split -> Train (with oversampling): {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    if config.SINGLE_LABEL_MODE:
        print(f"Class Weights: None (Dataset is balanced at {config.SAMPLES_PER_CLASS} per class)")
    else:
        train_labels_unique = unique_df[unique_df['Image_ID'].isin(train_ids)][TARGET_CLASSES].values
        neg_counts = (train_labels_unique == 0).sum(axis=0)
        pos_counts = (train_labels_unique == 1).sum(axis=0)
        pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(DEVICE)
        print(f"Class Weights: { {c: f'{w:.2f}' for c, w in zip(TARGET_CLASSES, pos_weight.cpu().tolist())} }")

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
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=3)
    val_loader   = DataLoader(ChestXRayDataset(val_df, DATA_DIR, val_trans),
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=3)

    model = VGGSwinHybridNet(
        num_classes=NUM_CLASSES, 
        drop_path_rate=config.DROP_PATH_RATE, 
        head_dropout=config.HEAD_DROPOUT
    ).to(DEVICE)
    if hasattr(model.swin_model, 'set_grad_checkpointing'):
        model.swin_model.set_grad_checkpointing(enable=True)


    if config.SINGLE_LABEL_MODE:
        def criterion(inputs, targets):
            smooth = targets * (1 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / NUM_CLASSES)
            return torch.nn.functional.cross_entropy(inputs, smooth)
    else:
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        def criterion(inputs, targets):
            smooth = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            return bce(inputs, smooth)

    # Phase schedule: (name, epochs, lr, freeze_backbone, freeze_swin)
    phases = [
        ("Warmup",  5,  1e-3, True,  True),
        ("Phase_1", 20, config.LR, True,  False),
        ("Phase_2", 15, config.LR, False, False),
    ]

    best_val_auc = 0.0
    best_val_f1  = 0.0
    start_phase_idx = 0
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        best_val_auc    = ckpt.get('best_val_auc', 0.0)
        best_val_f1     = ckpt.get('best_val_f1', 0.0)
        start_phase_idx = ckpt.get('phase_idx', 0)
        start_epoch     = ckpt.get('epoch', 0) + 1
        if start_epoch > phases[start_phase_idx][1]:
            start_phase_idx += 1
            start_epoch = 1

    for phase_idx in range(start_phase_idx, len(phases)):
        phase_name, epochs, lr, freeze_backbone, freeze_swin = phases[phase_idx]
        print(f"\n>>> {phase_name} (LR={lr})")

        for p in model.backbone.parameters():    p.requires_grad = not freeze_backbone
        for p in model.swin_layers.parameters(): p.requires_grad = not freeze_swin

        if phase_name == "Phase_2":
            param_groups = [
                {'params': model.head.parameters(),       'lr': lr},
                {'params': model.se.parameters(),         'lr': lr},
                {'params': model.bridge.parameters(),     'lr': lr},
                {'params': model.swin_norm.parameters(),  'lr': lr},
                {'params': model.swin_layers.parameters(),'lr': lr * 0.1},
                {'params': model.backbone.parameters(),   'lr': lr * 0.1},
            ]
            valid_groups = [{'params': [p for p in g['params'] if p.requires_grad], 'lr': g['lr']}
                            for g in param_groups]
            optimizer_class = getattr(torch.optim, getattr(config, "OPTIMIZER", "AdamW"))
            optimizer = optimizer_class(valid_groups, weight_decay=config.WEIGHT_DECAY)
        else:
            optimizer_class = getattr(torch.optim, getattr(config, "OPTIMIZER", "AdamW"))
            optimizer = optimizer_class(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=config.WEIGHT_DECAY
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        if os.path.exists(checkpoint_path) and start_phase_idx == phase_idx and start_epoch > 1:
            try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception: print("Could not restore optimizer state.")

        current_start = start_epoch if phase_idx == start_phase_idx else 1
        epochs_no_improve = 0

        for epoch in range(current_start, epochs + 1):
            t0 = time.time()
            model.train()
            # ALWAYS keep VGG backbone in eval mode to freeze BatchNorm running stats.
            # Fine-tuning BN stats at this stage causes NaN explosions.
            model.backbone.eval()
            if freeze_swin:
                model.swin_layers.eval()

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
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            torch.save({'phase_idx': phase_idx, 'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_auc': best_val_auc, 'best_val_f1': best_val_f1},
                       checkpoint_path)

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n⏹ Early stopping (no AUC improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            print(f"Loaded best model (AUC={best_val_auc:.4f}) for next phase.")

    print(f"\n✅ Done. Best Val AUC={best_val_auc:.4f} | F1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
