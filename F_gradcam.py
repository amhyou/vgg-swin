import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import VGGSwinHybridNet
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import config

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR   = config.CHECKPOINT_DIR
ROI_DIR     = config.ROI_IMAGE_DIR      # model was trained on these
RAW_DIR     = config.RAW_IMAGE_DIR      # original images for overlay display
METADATA_PATH = config.METADATA_PATH
OUTPUT_DIR  = os.path.join(config.RESULTS_DIR, "gradcam")
IMG_SIZE    = config.IMG_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_IMAGES_PER_CLASS = 5

TARGET_CLASSES = config.TARGET_CLASSES
NUM_CLASSES = config.NUM_CLASSES
print(f"-> {NUM_CLASSES}-Class Dataset: {TARGET_CLASSES}")


def get_image_selection():
    """Pick representative images per class (pure single-label preferred)."""
    df = pd.read_csv(METADATA_PATH).drop_duplicates(subset='Image_ID')
    selection = {}
    for cls in TARGET_CLASSES:
        pos = df[df[cls] == 1]
        pure = pos[pos[TARGET_CLASSES].sum(axis=1) == 1] if cls != 'Normal' else pos
        candidates = pure if len(pure) >= NUM_IMAGES_PER_CLASS else pos
        sampled = candidates.sample(n=min(NUM_IMAGES_PER_CLASS, len(candidates)), random_state=42)
        selection[cls] = sampled['Image_ID'].tolist()
    return selection


class SerialGradCAM:
    """
    Dual Grad-CAM for the serial VGG→Swin model:
      - CNN hook  → last Conv2d in VGG backbone  (before final MaxPool)
      - Swin hook → swin_norm output             (after all transformer blocks)
    """
    def __init__(self, model):
        self.model = model
        self.cnn_act = self.cnn_grad = None
        self.swin_act = self.swin_grad = None

        # Hook the last Conv2d in VGG backbone (features[-3] = Conv2d before MaxPool)
        cnn_target = None
        for layer in reversed(list(model.backbone.children())):
            if isinstance(layer, nn.Conv2d):
                cnn_target = layer
                break
        cnn_target.register_forward_hook(self._save_cnn_act)
        cnn_target.register_full_backward_hook(self._save_cnn_grad)

        # Hook swin_norm output
        model.swin_norm.register_forward_hook(self._save_swin_act)
        model.swin_norm.register_full_backward_hook(self._save_swin_grad)

    def _save_cnn_act(self, m, i, o):   self.cnn_act = o
    def _save_cnn_grad(self, m, gi, go): self.cnn_grad = go[0]
    def _save_swin_act(self, m, i, o):   self.swin_act = o
    def _save_swin_grad(self, m, gi, go): self.swin_grad = go[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        outputs[0, class_idx].backward()

        # ── CNN heatmap [B, C, H, W] → [H, W] ──────────────────────────
        weights = self.cnn_grad.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        cnn_hm  = (weights * self.cnn_act).sum(dim=1).squeeze()  # [H, W]
        cnn_hm  = F.relu(cnn_hm)
        cnn_hm  = cnn_hm / (cnn_hm.max() + 1e-8)

        # ── Swin heatmap [B, H, W, C] → [H, W] ──────────────────────────
        # In newer timm Swin, output is 4D spatial: [B, H, W, C]
        token_w  = self.swin_grad.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, C]
        swin_hm  = (token_w * self.swin_act).sum(dim=-1).squeeze() # [H, W]
        swin_hm  = F.relu(swin_hm)
        swin_hm  = swin_hm / (swin_hm.max() + 1e-8)

        return cnn_hm.detach().cpu().numpy(), swin_hm.detach().cpu().numpy()


def create_viz(roi_path, raw_path, cnn_hm, swin_hm, class_name, save_path):
    """
    Three-panel figure:
      Left  → Original raw X-ray (what a clinician sees)
      Middle → CNN Grad-CAM overlay on original
      Right  → Swin Grad-CAM overlay on original
    The model ran on the ROI image; the heatmap is overlaid on the raw image.
    """
    # Load original for display (fall back to ROI if raw not available)
    original = cv2.imread(raw_path) if (raw_path and os.path.exists(raw_path)) else cv2.imread(roi_path)
    if original is None:
        print(f"  Warning: could not load image for {class_name}. Skipping.")
        return
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    def make_overlay(hm):
        hm_resized = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
        color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
        return cv2.addWeighted(original, 0.6, color, 0.4, 0)

    cnn_overlay  = make_overlay(cnn_hm)
    swin_overlay = make_overlay(swin_hm)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original,     cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original X-Ray",       fontsize=14, fontweight='bold')
    axes[1].imshow(cv2.cvtColor(cnn_overlay,  cv2.COLOR_BGR2RGB))
    axes[1].set_title("CNN Focus (VGG16)",    fontsize=14, fontweight='bold')
    axes[2].imshow(cv2.cvtColor(swin_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Transformer Focus (Swin)", fontsize=14, fontweight='bold')
    for ax in axes: ax.axis('off')

    fig.suptitle(f"Dual Grad-CAM — {class_name}", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fold_paths  = sorted(glob.glob(f"{MODEL_DIR}/best_model_fold*.pth"))
    single_path = f"{MODEL_DIR}/best_model.pth"
    model_path  = fold_paths[0] if fold_paths else single_path
    print(f"Loading model: {model_path}")

    model = VGGSwinHybridNet(
        num_classes=NUM_CLASSES,
        drop_path_rate=config.DROP_PATH_RATE,
        head_dropout=config.HEAD_DROPOUT
    ).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: {model_path} not found. Using random weights (for testing only).")
    model.eval()

    cam = SerialGradCAM(model)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    raw_available = os.path.exists(RAW_DIR)
    if not raw_available:
        print(f"Warning: RAW_IMAGE_DIR '{RAW_DIR}' not found. "
              f"Overlays will use ROI images instead. "
              f"Download the raw dataset to '{RAW_DIR}' for proper visualizations.")

    image_selection = get_image_selection()

    for cls, img_names in image_selection.items():
        class_idx = TARGET_CLASSES.index(cls)
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

        for img_name in img_names:
            roi_path = os.path.join(ROI_DIR, img_name)
            raw_path = os.path.join(RAW_DIR, img_name) if raw_available else None

            if not os.path.exists(roi_path):
                print(f"  Skipping {img_name} (ROI not found)")
                continue

            # Model input = ROI image
            input_tensor = transform(Image.open(roi_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                cnn_hm, swin_hm = cam.generate(input_tensor, class_idx)

            # Visualization display = original raw image
            stem = img_name.replace('.png', '').replace('.jpg', '')
            save_path = os.path.join(OUTPUT_DIR, cls, f"gradcam_{cls}_{stem}.png")
            create_viz(roi_path, raw_path, cnn_hm, swin_hm, cls, save_path)
            print(f"  ✓ {cls} | {img_name}")

    print(f"\nGrad-CAM complete. Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
