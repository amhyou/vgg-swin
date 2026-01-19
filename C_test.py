import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from models.main import VGGSwinHybridNet
from B_train import ChestXRayDataset
from torchvision import transforms
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# --- CONFIGURATION ---
DATA_DIR = "data/roi"
METADATA_PATH = "metadata/DATA_ROI.csv"
MODEL_PATH = "weights/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Normal']
N_BOOTSTRAP = 1000

def plot_confusion_matrix(y_true, y_pred):
    """Generates a per-class confusion matrix for multi-label analysis."""
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, (matrix, name) in enumerate(zip(mcm, TARGET_CLASSES)):
        sns.heatmap(matrix, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrices.png')
    plt.close()

def calculate_auc_ci(y_true, y_probs, n_bootstraps=N_BOOTSTRAP):
    """Calculates 95% CI for AUC using bootstrapping."""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue # Skip if only one class is present
        score = roc_auc_score(y_true[indices], y_probs[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_bound = sorted_scores[int(0.025 * len(sorted_scores))]
    upper_bound = sorted_scores[int(0.975 * len(sorted_scores))]
    return lower_bound, upper_bound

def plot_calibration_curves(y_true, y_probs):
    """Plots calibration curves for each class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, name in enumerate(TARGET_CLASSES):
        prob_true, prob_pred = calibration_curve(y_true[:, i], y_probs[:, i], n_bins=10, strategy='uniform')
        axes[i].plot(prob_pred, prob_true, marker='o', label=name)
        axes[i].plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        axes[i].set_title(f'Calibration Curve: {name}')
        axes[i].set_xlabel('Mean Predicted Probability')
        axes[i].set_ylabel('Fraction of Positives')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('results/test_calibration_curves.png')
    plt.close()

def measure_inference_speed(model, loader, device):
    """Measures average inference time per image."""
    model.eval()
    total_time = 0
    total_images = 0
    
    # Warm-up GPU
    if device.type == 'cuda':
        for _ in range(5):
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            _ = model(dummy_input)
            torch.cuda.synchronize()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            start_time = time.time()
            _ = model(images)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_images += images.size(0)
            
    return total_time / total_images

def main():
    if not os.path.exists('results'): os.makedirs('results')

    # 1. Load Test Set
    full_df = pd.read_csv(METADATA_PATH)
    _, temp_df = train_test_split(full_df, test_size=0.3, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    test_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_loader = DataLoader(ChestXRayDataset(test_df, DATA_DIR, test_trans), batch_size=32)

    # 2. Load Model
    model = VGGSwinHybridNet(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 3. Inference
    all_labels, all_probs = [], []
    print("Running Final Test Evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            all_probs.append(torch.sigmoid(outputs).cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (all_probs > 0.5).astype(float)

    # 4. Save Quantitative Results
    report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv('results/final_test_report.csv')
    
    # --- Advanced Metrics ---
    print("\n--- Advanced Metrics ---")
    
    # Per-class AUC with Confidence Intervals
    auc_results = {}
    for i, cls in enumerate(TARGET_CLASSES):
        auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        lower, upper = calculate_auc_ci(all_labels[:, i], all_probs[:, i])
        auc_results[cls] = f"{auc_score:.4f} (95% CI: {lower:.4f}-{upper:.4f})"
    print("Final Test AUC Scores (with 95% CI):")
    for cls, result in auc_results.items():
        print(f"- {cls}: {result}")

    # Inference Speed
    avg_inference_time = measure_inference_speed(model, test_loader, DEVICE)
    print(f"\nAverage Inference Time: {avg_inference_time * 1000:.2f} ms per image")

    # 5. Visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(all_labels, all_preds)
    plot_calibration_curves(all_labels, all_probs)
    print("Results saved to 'results/' directory.")

if __name__ == "__main__":
    main()