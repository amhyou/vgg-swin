import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import config
from model import VGGSwinHybridNet
from C_train import ChestXRayDataset, mixup_batch, evaluate, TARGET_CLASSES, NUM_CLASSES, DEVICE

# ─── HPO CONFIG ──────────────────────────────────────────────────────────────
PROXY_MODEL = 'swin_tiny_patch4_window7_224'
PROXY_IMG_SIZE = 224
DATA_SUBSET_FRAC = 0.2 # Use 20% of data to speed up epochs
EPOCHS_PER_TRIAL = 10   # Only run 10 epochs max to evaluate a hyperparameter set
N_TRIALS = 100           # Total number of configurations to test

DATA_DIR = config.ROI_IMAGE_DIR
METADATA_PATH = config.METADATA_PATH

# ─── DATA PREPARATION ────────────────────────────────────────────────────────
def prepare_hpo_dataloaders(batch_size):
    full_df = pd.read_csv(METADATA_PATH)

    if config.SINGLE_LABEL_MODE:
        full_df['Total_Labels'] = full_df[TARGET_CLASSES].sum(axis=1)
        full_df = full_df[full_df['Total_Labels'] == 1].copy()
        full_df.drop(columns=['Total_Labels'], inplace=True)
    elif config.NUM_CLASSES == 2:
        all_diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal']
        if 'Pneumonia' in full_df.columns: all_diseases.append('Pneumonia')
        full_df['disease_sum'] = full_df[all_diseases].sum(axis=1)
        mask = (full_df['disease_sum'] == 1) & ((full_df['Normal'] == 1) | (full_df['Effusion'] == 1))
        full_df = full_df[mask].copy()
        full_df.drop(columns=['disease_sum'], inplace=True)

    def get_patient_id(img_id):
        if img_id.startswith('nih_'): return img_id.split('_')[1]
        if 'patient' in img_id:
            for p in img_id.split('_'):
                if 'patient' in p: return p
        return img_id

    full_df['Patient_Group'] = full_df['Image_ID'].apply(get_patient_id)
    unique_df = full_df.drop_duplicates(subset='Image_ID').copy()

    # Subsample data for proxy run
    gss_sub = GroupShuffleSplit(n_splits=1, train_size=DATA_SUBSET_FRAC, random_state=42)
    subset_idx, _ = next(gss_sub.split(unique_df, groups=unique_df['Patient_Group']))
    subset_unique_df = unique_df.iloc[subset_idx]
    
    # Train / Val Split (80/20 on the subset)
    gss_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_split.split(subset_unique_df, groups=subset_unique_df['Patient_Group']))
    
    train_ids = set(subset_unique_df.iloc[train_idx]['Image_ID'])
    val_ids   = set(subset_unique_df.iloc[val_idx]['Image_ID'])
    
    # Val set strictly unique
    val_df   = subset_unique_df[subset_unique_df['Image_ID'].isin(val_ids)].copy()

    if config.SINGLE_LABEL_MODE:
        base_train_df = subset_unique_df[subset_unique_df['Image_ID'].isin(train_ids)].copy()
        target_samples = int(config.SAMPLES_PER_CLASS * DATA_SUBSET_FRAC)
        sampled_dfs = []
        for cls in TARGET_CLASSES:
            cls_df = base_train_df[base_train_df[cls] == 1]
            if len(cls_df) >= target_samples:
                sampled_dfs.append(cls_df.sample(n=target_samples, replace=False, random_state=42))
            elif len(cls_df) > 0:
                sampled_dfs.append(cls_df.sample(n=target_samples, replace=True, random_state=42))
        train_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        train_df = full_df[full_df['Image_ID'].isin(train_ids)].copy()

    # Determine class weights
    if config.SINGLE_LABEL_MODE:
        pos_weight = None
    else:
        train_labels = subset_unique_df[subset_unique_df['Image_ID'].isin(train_ids)][TARGET_CLASSES].values
        neg_counts = (train_labels == 0).sum(axis=0)
        pos_counts = (train_labels == 1).sum(axis=0)
        pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(DEVICE)

    # Note: Using PROXY_IMG_SIZE (224)
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(PROXY_IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        transforms.Resize((PROXY_IMG_SIZE, PROXY_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    NUM_WORKERS = min(os.cpu_count() or 4, 8)
    train_loader = DataLoader(ChestXRayDataset(train_df, DATA_DIR, train_trans),
                              batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(ChestXRayDataset(val_df, DATA_DIR, val_trans),
                              batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
                              
    return train_loader, val_loader, pos_weight


# ─── CUSTOM LOSS ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()


# ─── OPTUNA OBJECTIVE ────────────────────────────────────────────────────────
def objective(trial):
    # 1. Suggest Hyperparameters (Focusing ONLY on the Top 5 most important)
    lr              = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay    = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    mixup_alpha     = trial.suggest_float("mixup_alpha", 0.1, 0.8)
    drop_path_rate  = trial.suggest_float("drop_path_rate", 0.1, 0.3)
    optimizer_name  = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    
    # Hardcoded defaults for hyperparameters with < 1% importance
    label_smoothing = 0.1
    head_dropout    = 0.3
    loss_name       = "BCE"
    
    # 224x224 takes way less memory, so we can use a larger batch size for faster epochs.
    batch_size = 64
    
    # 2. Setup Data
    train_loader, val_loader, pos_weight = prepare_hpo_dataloaders(batch_size=batch_size)
    
    # 3. Setup Proxy Model
    model = VGGSwinHybridNet(
        num_classes=NUM_CLASSES,
        swin_model_name=PROXY_MODEL,
        drop_path_rate=drop_path_rate,
        head_dropout=head_dropout
    ).to(DEVICE)
    
    if config.SINGLE_LABEL_MODE:
        def criterion(inputs, targets):
            smooth = targets * (1 - label_smoothing) + (label_smoothing / NUM_CLASSES)
            return torch.nn.functional.cross_entropy(inputs, smooth)
    else:
        if loss_name == "BCE":
            base_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            base_criterion = FocalLoss(pos_weight=pos_weight, gamma=2.0)
            
        def criterion(inputs, targets):
            smooth = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            return base_criterion(inputs, smooth)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Always use CosineAnnealing (it is vastly superior to StepLR for Vision Transformers)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS_PER_TRIAL, T_mult=1)
        
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # 4. Fast Training Loop (No phase freezing to test pure capacity fast)
    best_auc = 0.0
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        model.train()
        
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Trial {trial.number}] Epoch {epoch}/{EPOCHS_PER_TRIAL}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = mixup_batch(images, labels, alpha=mixup_alpha)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{train_loss/(pbar.n+1):.4f}"})

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion)
        current_auc = val_metrics['auc_macro']
        best_auc = max(best_auc, current_auc)
        
        print(f"  -> Trial {trial.number} | Epoch {epoch}/{EPOCHS_PER_TRIAL} | Val AUC: {current_auc:.4f} (Best: {best_auc:.4f})")
        
        scheduler.step()
        
        # Report intermediate objective value to Pruner
        trial.report(current_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_auc


if __name__ == "__main__":
    print(f"Starting Optuna HPO with Proxy Model: {PROXY_MODEL} ({PROXY_IMG_SIZE}x{PROXY_IMG_SIZE})")
    
    # Create study using MedianPruner and SQLite database for persistence
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(config.RESULTS_DIR, "optuna_study.db")
    
    study = optuna.create_study(
        study_name="vgg_swin_hpo_v2",
        direction="maximize", 
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=pruner
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Showing best results so far...")

    print("\n" + "="*50)
    print("HPO Complete!")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (AUC): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
    
    # Optionally save results to CSV
    df = study.trials_dataframe()
    df.to_csv(f"{config.RESULTS_DIR}/hpo_optuna_results.csv", index=False)
    print(f"Full results saved to {config.RESULTS_DIR}/hpo_optuna_results.csv")
    
    # ─── GENERATE VISUALIZATIONS FOR THESIS ──────────────────────────────────
    print("\nGenerating interactive HTML plots for your thesis...")
    try:
        import optuna.visualization as vis
        
        # Generate the plots
        fig_history = vis.plot_optimization_history(study)
        fig_importances = vis.plot_param_importances(study)
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_slice = vis.plot_slice(study)
        
        # Save them as interactive HTML files
        fig_history.write_html(f"{config.RESULTS_DIR}/hpo_optimization_history.html")
        fig_importances.write_html(f"{config.RESULTS_DIR}/hpo_param_importances.html")
        fig_parallel.write_html(f"{config.RESULTS_DIR}/hpo_parallel_coordinate.html")
        fig_slice.write_html(f"{config.RESULTS_DIR}/hpo_slice.html")
        
        print(f"✅ Successfully saved 4 interactive visualization plots to {config.RESULTS_DIR}/")
    except ImportError:
        print("⚠️ Plotly is not installed. To generate thesis plots, run: pip install plotly")
        print("After installing, you can generate the plots anytime from the saved SQLite database.")
