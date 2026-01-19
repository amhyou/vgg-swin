import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.main import VGGSwinHybridNet
from PIL import Image
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = "weights/best_model.pth"
RAW_DIR = "data/raw"             # Original data for background
ROI_DIR = "data/roi"             # Preprocessed data for model input
METADATA_PATH = "metadata/DATA_ROI.csv"
OUTPUT_DIR = "results/gradcam_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Normal']

# --- Define the exact images to visualize ---
IMAGE_SELECTION = {
    'Atelectasis': '00000005_005.png',
    'Cardiomegaly': '00000040_000.png',
    'Effusion': '00000077_000.png',
    'Pneumonia': '00000203_000.png',
    'Normal': '00000318_004.png'
}


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Capture gradients and activations from the final VGG base layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        loss = output[0, class_idx]
        loss.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Keep only positive contributions (ReLU)
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.detach().cpu().numpy()

def create_viz(raw_path, roi_path, heatmap, class_name, save_path):
    # 1. Load the original raw image for the viewer
    original = cv2.imread(raw_path)
    original = cv2.resize(original, (224, 224))
    
    # 2. Load the isolated lung for reference
    roi_img = cv2.imread(roi_path)
    roi_img = cv2.resize(roi_img, (224, 224))
    
    # 3. Process Heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay heatmap on the ORIGINAL image
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    
    # Create a 3-panel comparison for your thesis 
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.title("Raw X-Ray")
    plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)); plt.title("Isolated ROI")
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    model = VGGSwinHybridNet(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # --- Disable inplace ReLU in the VGG backbone ---
    for module in model.backbone.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    
    # Target final conv layer for maximum spatial detail
    cam = GradCAM(model, model.backbone[15])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Loop through images ---
    for cls, img_name in IMAGE_SELECTION.items():
        class_idx = TARGET_CLASSES.index(cls)
        
        raw_path = os.path.join(RAW_DIR, img_name)
        roi_path = os.path.join(ROI_DIR, img_name)
        
        if not os.path.exists(raw_path) or not os.path.exists(roi_path):
            print(f"Warning: Could not find {img_name} for class {cls}. Skipping.")
            continue
        
        # Generate heatmap using the ROI image
        input_tensor = transform(Image.open(roi_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        heatmap = cam.generate_heatmap(input_tensor, class_idx)
        
        create_viz(raw_path, roi_path, heatmap, cls, os.path.join(OUTPUT_DIR, f"thesis_gradcam_{cls}.png"))
        print(f"Generated clean visualization for {cls} using {img_name}")

if __name__ == "__main__":
    main()