import os
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.main import VGGSwinHybridNet
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = "data/roi"
METADATA_PATH = "metadata/DATA_ROI.csv"
LOG_FILE = "results/comprehensive_training_log.csv"
CHECKPOINT_DIR = "weights"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Normal']

class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        labels = torch.tensor(row[TARGET_CLASSES].values.astype(float), dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, labels

def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (all_probs > 0.5).astype(float)
    
    # --- METRICS CALCULATIONS ---
    # Calculate label-based accuracy (intuitive metric)
    # This checks every single label prediction
    label_acc = np.mean(all_labels == all_preds)

    metrics = {
        'loss': val_loss / len(loader),
        'acc_label_based': label_acc, # The intuitive accuracy
        'acc_subset': accuracy_score(all_labels, all_preds), # The strict, all-or-nothing accuracy
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'auc_macro': roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    }
    
    # Per-Class Metrics
    for i, class_name in enumerate(TARGET_CLASSES):
        metrics[f'f1_{class_name}'] = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        metrics[f'auc_{class_name}'] = roc_auc_score(all_labels[:, i], all_probs[:, i])

    return metrics

def log_to_csv(data_dict):
    df = pd.DataFrame([data_dict])
    if not os.path.isfile(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

def main():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    
    # 1. Split & Load (70/15/15 Patient-Level Split)
    full_df = pd.read_csv(METADATA_PATH)
    train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ChestXRayDataset(train_df, DATA_DIR, train_trans), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ChestXRayDataset(val_df, DATA_DIR, val_trans), batch_size=BATCH_SIZE)

    # 2. Model, Optimizer, Loss
    model = VGGSwinHybridNet(num_classes=5).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    phases = [
        ("Phase_1", 15, 1e-4, True),
        ("Phase_2", 30, 1e-5, False)
    ]

    best_val_f1 = 0
    
    for phase_name, epochs, lr, freeze_vgg in phases:
        print(f"\n>>> STARTING {phase_name} (LR: {lr})")
        
        for param in model.backbone.parameters():
            param.requires_grad = not freeze_vgg
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            model.train()
            train_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_metrics = evaluate(model, val_loader, criterion)
            epoch_time = time.time() - start_time
            
            log_entry = {
                'Phase': phase_name,
                'Epoch': epoch,
                'LR': lr,
                'Time_Sec': epoch_time,
                'Train_Loss': train_loss / len(train_loader),
                **{f'Val_{k}': v for k, v in val_metrics.items()}
            }
            
            log_to_csv(log_entry)
            print(f"[{phase_name} E{epoch}] Val Loss: {val_metrics['loss']:.4f} | Subset Acc: {val_metrics['acc_subset']:.4f} | Label Acc: {val_metrics['acc_label_based']:.4f} | F1: {val_metrics['f1_macro']:.4f} | AUC: {val_metrics['auc_macro']:.4f}")

            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
                print("--- New Best Model Saved ---")

if __name__ == "__main__":
    main()