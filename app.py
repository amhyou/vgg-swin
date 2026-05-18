import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import plotly.express as px
import pandas as pd
import os

# Set layout early
st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide", page_icon="🩺")

# Import internal modules
try:
    import config
    from unet import get_unet
    from model import VGGSwinHybridNet
    from F_gradcam import SerialGradCAM
except ImportError as e:
    st.error(f"Error importing required modules. Please run this app from the project root directory. Details: {e}")
    st.stop()

# ==========================================
# Configuration & Constants
# ==========================================
IMG_SIZE = config.IMG_SIZE
TARGET_CLASSES = config.TARGET_CLASSES
NUM_CLASSES = config.NUM_CLASSES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Caching Model Loading
# ==========================================
@st.cache_resource
def load_unet_model(weights_path):
    """Loads the U-Net model and weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"U-Net weights not found at: {weights_path}")
    model = get_unet((IMG_SIZE, IMG_SIZE, 1))
    model.load_weights(weights_path)
    return model

@st.cache_resource
def load_vggswin_model(weights_path):
    """Loads the VGGSwinHybridNet model and weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"VGG-Swin weights not found at: {weights_path}")
    
    model = VGGSwinHybridNet(
        num_classes=NUM_CLASSES,
        drop_path_rate=config.DROP_PATH_RATE,
        head_dropout=config.HEAD_DROPOUT
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ==========================================
# Main App Layout
# ==========================================
def main():
    st.title("🩺 VGG-Swin HybridNet Chest X-Ray Classifier")
    st.markdown(
        f"**Multi-Class Detection Pipeline:** {', '.join(TARGET_CLASSES)}"
    )
    
    # ------------------------------------------
    # 1. Sidebar Controls
    # ------------------------------------------
    st.sidebar.header("Controls & Configuration")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload a Chest X-Ray Image", 
        type=["png", "jpg", "jpeg"]
    )
    
    mock_mode = st.sidebar.checkbox("Enable Mock Mode (No GPU/Weights required)", value=False)
    
    st.sidebar.subheader("Model Weights")
    unet_weights_path = st.sidebar.text_input(
        "U-Net Weights Path", 
        value="weights/cxr_reg_weights.best.hdf5"
    )
    vggswin_weights_path = st.sidebar.text_input(
        "VGG-Swin Weights Path", 
        value="weights/best_model.pth"
    )
    
    threshold = st.sidebar.slider(
        "Classification Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    if not uploaded_file:
        st.info("👈 Please upload a Chest X-Ray image from the sidebar to begin analysis.")
        return
        
    if not mock_mode:
        try:
            unet_model = load_unet_model(unet_weights_path)
            vggswin_model = load_vggswin_model(vggswin_weights_path)
        except Exception as e:
            st.error(f"Error loading models. Please verify the weight paths. \n\nDetails: {e}")
            return

    # ------------------------------------------
    # 2. Step 1: Input & Preprocessing
    # ------------------------------------------
    st.header("Step 1: Input & Preprocessing")
    
    # Load the raw image
    raw_pil = Image.open(uploaded_file).convert('RGB')
    raw_np = np.array(raw_pil)
    
    # Resize to 224x224 as required by the models
    raw_resized = cv2.resize(raw_np, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(raw_resized, cv2.COLOR_RGB2GRAY)
    
    st.image(raw_resized, caption="Original Raw Chest X-Ray", width=350)
    
    if mock_mode:
        # Mock U-Net Lung Mask: A simple white ellipse on black background
        binary_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.ellipse(binary_mask, (IMG_SIZE//2, IMG_SIZE//2), (int(IMG_SIZE/3), int(IMG_SIZE/2.2)), 0, 0, 360, 1, -1)
        
        # Mock CLAHE Enhanced ROI: Just use the original image
        enhanced_roi = raw_resized.copy()
    else:
        # U-Net Lung Mask Extraction
        unet_input = (gray.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
        mask_pred = unet_model.predict(unet_input, verbose=0)[0, ..., 0]
        binary_mask = (mask_pred > 0.5).astype(np.uint8)
        
        # Apply bitwise mask to the original grayscale image
        masked_gray = cv2.bitwise_and(gray, gray, mask=binary_mask)
        
        # Apply CLAHE on the masked image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(masked_gray)
        
        # Convert back to RGB for display and for the VGG-Swin model
        enhanced_roi = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    # Display preprocessing results side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(binary_mask * 255, caption="Binary Lung Mask", use_container_width=True, clamp=True)
    with col2:
        st.image(enhanced_roi, caption="CLAHE Enhanced ROI", use_container_width=True)
        
    # ------------------------------------------
    # 3. Step 2: Inference
    # ------------------------------------------
    st.header("Step 2: Inference")
    
    if mock_mode:
        probs = np.random.rand(NUM_CLASSES)
    else:
        # Preprocess for PyTorch
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert enhanced ROI to tensor and run inference
        input_tensor = transform(enhanced_roi).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = vggswin_model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    # Prepare results for visualization
    results_df = pd.DataFrame({
        "Disease": TARGET_CLASSES,
        "Probability": probs
    })
    results_df["Prediction"] = results_df["Probability"].apply(
        lambda p: "Positive" if p >= threshold else "Negative"
    )
    
    # Visually appealing bar chart using Plotly
    fig = px.bar(
        results_df,
        x="Disease",
        y="Probability",
        color="Prediction",
        color_discrete_map={"Positive": "#ef4444", "Negative": "#3b82f6"},
        text=results_df["Probability"].apply(lambda x: f"{x:.1%}"),
        title="Disease Classification Probabilities"
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
    fig.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="",
        yaxis_title="Probability",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------------------------
    # 4. Step 3: Interpretability (Dual Grad-CAM)
    # ------------------------------------------
    st.header("Step 3: Interpretability (Dual Grad-CAM)")
    
    highest_class_idx = np.argmax(probs)
    highest_class_name = TARGET_CLASSES[highest_class_idx]
    highest_prob = probs[highest_class_idx]
    
    st.markdown(f"Generating heatmap for the most probable condition: **{highest_class_name}** ({highest_prob:.1%})")
    
    if mock_mode:
        # Generate random smooth heatmaps for mock mode
        cnn_hm = cv2.GaussianBlur(np.random.rand(28, 28).astype(np.float32), (5, 5), 0)
        swin_hm = cv2.GaussianBlur(np.random.rand(14, 14).astype(np.float32), (5, 5), 0)
        cnn_hm = cnn_hm / (cnn_hm.max() + 1e-8)
        swin_hm = swin_hm / (swin_hm.max() + 1e-8)
    else:
        # Instantiate Grad-CAM
        cam = SerialGradCAM(vggswin_model)
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            cnn_hm, swin_hm = cam.generate(input_tensor, highest_class_idx)
        
    def create_overlay(hm, original_rgb):
        """Resizes heatmap, applies jet colormap, and blends with original image."""
        hm_resized = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
        
        # Apply colormap (cv2 returns BGR)
        color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
        
        # Convert BGR to RGB for Streamlit/Matplotlib
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        # Blend original with heatmap
        overlay = cv2.addWeighted(original_rgb, 0.6, color_rgb, 0.4, 0)
        return overlay

    # Create overlays directly on the raw original image (resized)
    cnn_overlay = create_overlay(cnn_hm, raw_resized)
    swin_overlay = create_overlay(swin_hm, raw_resized)
    
    # Display side-by-side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(raw_resized, caption="Original Image", use_container_width=True)
    with col2:
        st.image(cnn_overlay, caption="CNN Focus (VGG16)", use_container_width=True)
    with col3:
        st.image(swin_overlay, caption="Transformer Focus (Swin)", use_container_width=True)

if __name__ == "__main__":
    main()
