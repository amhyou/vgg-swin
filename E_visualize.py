import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

# --- CONFIGURATION ---
LOG_FILE    = config.LOG_FILE
RESULTS_DIR = config.RESULTS_DIR

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Load the log created by train.py
try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print(f"Error: {LOG_FILE} not found. Please run training first.")
    exit()

total_epochs = len(df)
phase1_break = df[df['Phase'] == 'Warmup'].shape[0] if 'Warmup' in df['Phase'].values else 0
phase2_break = phase1_break + df[df['Phase'] == 'Phase_1'].shape[0] if 'Phase_1' in df['Phase'].values else 0
best_epoch = df['Val_f1_macro'].idxmax() + 1 if 'Val_f1_macro' in df.columns else None

def save_thesis_plot(title, filename, y_label="Metric Value", show_best=False):
    sns.set_theme(style="whitegrid", context="paper")
    plt.title(title, fontsize=14, fontweight='bold')
    
    if phase1_break > 0:
        plt.axvline(x=phase1_break, color='gray', linestyle='--', alpha=0.5, label='Phase 1 Start')
    if phase2_break > phase1_break:
        plt.axvline(x=phase2_break, color='purple', linestyle='--', alpha=0.5, label='Phase 2 Start')
    if show_best and best_epoch is not None:
        plt.axvline(x=best_epoch, color='green', linestyle='-', alpha=0.7, label=f'Best Epoch ({best_epoch})')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 1. Loss Curve (Standard and Log Scale)
plt.figure(figsize=(10, 6))
plt.plot(range(1, total_epochs + 1), df['Train_Loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(range(1, total_epochs + 1), df['Val_loss'], label='Validation Loss', color='orange', linewidth=2)
save_thesis_plot(f'Training vs. Validation Loss ({config.IMG_SIZE}x{config.IMG_SIZE})', 'loss_curve.png', y_label="Loss")

# 2. Accuracy
plt.figure(figsize=(10, 6))
if 'Val_accuracy' in df.columns:
    plt.plot(range(1, total_epochs + 1), df['Val_accuracy'], label='Accuracy', color='green', linewidth=2)
    plt.ylim(0, 1.0)
    save_thesis_plot(f'Accuracy Progression ({config.IMG_SIZE}x{config.IMG_SIZE})', 'accuracy_progression.png', y_label="Accuracy")

# 3. Key Performance Metrics (F1 & AUC & MCC)
plt.figure(figsize=(10, 6))
plt.plot(range(1, total_epochs + 1), df['Val_f1_macro'], label='Macro F1-Score', color='purple')
plt.plot(range(1, total_epochs + 1), df['Val_auc_macro'], label='Macro AUC', color='brown')
if 'Val_mcc' in df.columns:
    plt.plot(range(1, total_epochs + 1), df['Val_mcc'], label='MCC', color='teal')
if 'Val_kappa' in df.columns:
    plt.plot(range(1, total_epochs + 1), df['Val_kappa'], label="Cohen's Kappa", color='navy')
plt.ylim(0, 1.0)
save_thesis_plot(f'Key Performance Metrics (Validation) ({config.IMG_SIZE}x{config.IMG_SIZE})', 'macro_metrics_trend.png', y_label="Score", show_best=True)

# 4. Per-Class F1 Progression
plt.figure(figsize=(12, 7))
# Dynamically extract classes from the columns
classes = [col.replace('Val_f1_', '') for col in df.columns if col.startswith('Val_f1_') and 'macro' not in col]
colors = sns.color_palette("husl", len(classes))
for cls, color in zip(classes, colors):
    plt.plot(range(1, total_epochs + 1), df[f'Val_f1_{cls}'], label=f'F1: {cls}', color=color)
save_thesis_plot(f'Per-Class F1-Score Improvement ({config.IMG_SIZE}x{config.IMG_SIZE})', 'per_class_f1.png', y_label="F1-Score")

print(f"Visualizations saved to '{RESULTS_DIR}/' directory.")
print("Key plots generated:")
print("- loss_curve.png")
print("- accuracy_progression.png")
print("- macro_metrics_trend.png")
print("- per_class_f1.png")
