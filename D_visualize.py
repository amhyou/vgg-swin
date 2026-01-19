import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
LOG_FILE = "results/comprehensive_training_log.csv"
RESULTS_DIR = "results"

# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Load the log created by train.py
df = pd.read_csv(LOG_FILE)
total_epochs = len(df)
phase_break = df[df['Phase'] == 'Phase_1'].shape[0] # Dynamically find phase break

def save_thesis_plot(title, filename, y_label="Metric Value"):
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axvline(x=phase_break, color='red', linestyle='--', alpha=0.7, label='Fine-Tuning Start')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

# 1. Loss Curve (Standard and Log Scale)
plt.figure(figsize=(10, 6))
plt.plot(range(1, total_epochs + 1), df['Train_Loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(range(1, total_epochs + 1), df['Val_loss'], label='Validation Loss', color='orange', linewidth=2)
save_thesis_plot('Training vs. Validation Loss', 'loss_curve.png', y_label="BCE Loss")

# 2. Accuracy Comparison (Label-Based vs. Subset)
# This plot directly addresses the "40% vs 97%" issue.
plt.figure(figsize=(10, 6))
plt.plot(range(1, total_epochs + 1), df['Val_acc_label_based'], label='Label-Based Accuracy (Intuitive)', color='green', linewidth=2)
plt.plot(range(1, total_epochs + 1), df['Val_acc_subset'], label='Subset Accuracy (Strict)', color='red', linestyle='--', linewidth=2)
plt.ylim(0, 1.0) # Set y-axis from 0 to 1 for accuracy
save_thesis_plot('Accuracy Comparison: Label-Based vs. Subset', 'accuracy_comparison.png', y_label="Accuracy")

# 3. Key Performance Metrics (F1 & AUC)
plt.figure(figsize=(10, 6))
plt.plot(range(1, total_epochs + 1), df['Val_f1_macro'], label='Macro F1-Score', color='purple')
plt.plot(range(1, total_epochs + 1), df['Val_auc_macro'], label='Macro AUC', color='brown')
plt.ylim(0, 1.0)
save_thesis_plot('Key Performance Metrics (Validation)', 'macro_metrics_trend.png', y_label="Score")

# 4. Per-Class F1 Progression
plt.figure(figsize=(12, 7))
classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Normal']
colors = sns.color_palette("husl", len(classes))
for cls, color in zip(classes, colors):
    if f'Val_f1_{cls}' in df.columns:
        plt.plot(range(1, total_epochs + 1), df[f'Val_f1_{cls}'], label=f'F1: {cls}', color=color)
save_thesis_plot('Per-Class F1-Score Improvement', 'per_class_f1.png', y_label="F1-Score")

print(f"Visualizations saved to '{RESULTS_DIR}/' directory.")
print("Key plots generated:")
print("- loss_curve.png")
print("- accuracy_comparison.png (Shows intuitive vs. strict accuracy)")
print("- macro_metrics_trend.png")
print("- per_class_f1.png")