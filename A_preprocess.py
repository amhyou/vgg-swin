import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.unet import get_unet

# --- CONFIGURATION ---
DATA_DIR = "data"
RAW_IMAGES = os.path.join(DATA_DIR, "raw")
ROI_OUTPUT = os.path.join(DATA_DIR, "roi")
METADATA_DIR = "metadata"
INPUT_CSV = os.path.join(METADATA_DIR, "Data_Entry_2017.csv")
FINAL_METADATA = os.path.join(METADATA_DIR, "DATA_ROI.csv")
WEIGHTS = "weights/cxr_reg_weights.best.hdf5"

# Targets: 4 Pathologies + Normal 
PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia']
TARGET_CLASSES = PATHOLOGIES + ['Normal']
TARGET_SAMPLES_PER_CLASS = 9000 


class LungProcessor:
    def __init__(self, weights_path, img_size=224):
        self.img_size = img_size

        # Build the U-Net architecture with input shape (img_size, img_size, 1) for grayscale images
        self.model = get_unet((img_size, img_size, 1))

        # Load the pre-trained weights into the model
        self.model.load_weights(weights_path)

        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, img_path):
        # Step 1: Load and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None  # Return None if image loading fails
        img = cv2.resize(img, (self.img_size, self.img_size))
        enhanced = self.clahe.apply(img)  # Apply CLAHE for contrast enhancement

        # Step 2: Prepare image for model prediction
        img_in = enhanced.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_in = np.expand_dims(img_in, axis=(0, -1))  # Add batch and channel dimensions

        # Step 3: Predict segmentation mask using the U-Net model
        mask = self.model.predict(img_in, verbose=0)[0]  # Get prediction, remove batch dim
        mask_binary = (mask > 0.5).astype(np.uint8)  # Threshold to create binary mask

        # Step 4: Apply mask to isolate lung regions
        return cv2.bitwise_and(enhanced, enhanced, mask=mask_binary)

    
def run_targeted_pipeline():
    if not os.path.exists(ROI_OUTPUT): os.makedirs(ROI_OUTPUT)
    if not os.path.exists(METADATA_DIR): os.makedirs(METADATA_DIR)

    print("Step 1: Filtering metadata for 5 classes (including Normal)...")
    df = pd.read_csv(INPUT_CSV)
    
    for cls in PATHOLOGIES:
        df[cls] = df['Finding Labels'].apply(lambda x: 1 if cls in x else 0)
    
    # Mapping 'No Finding' to Normal class 
    df['Normal'] = df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
    
    target_df = df[df[TARGET_CLASSES].sum(axis=1) > 0].copy()

    print("\nStep 2: Calculating class-aware balancing...")
    balanced_list = []
    unique_images_to_process = set()

    for cls in TARGET_CLASSES:
        subset = target_df[target_df[cls] == 1]
        count = len(subset)
        
        # Sampling down Normal if it exceeds the target; oversampling others if needed
        if count > TARGET_SAMPLES_PER_CLASS:
            subset = subset.sample(n=TARGET_SAMPLES_PER_CLASS, random_state=42)
            multiplier = 1
        else:
            multiplier = max(1, TARGET_SAMPLES_PER_CLASS // count)
        
        print(f" - {cls}: {len(subset)} base images. Augmentation factor: {multiplier}x")
        
        for i in range(multiplier):
            temp = subset.copy()
            temp['aug_instance'] = i # Tracking ID for Grad-CAM 
            balanced_list.append(temp)
            unique_images_to_process.update(subset['Image Index'].tolist())

    print(f"\nStep 3: ROI Isolation for new images (Skipping existing)...")
    processor = LungProcessor(WEIGHTS)
    
    for img_name in tqdm(list(unique_images_to_process)):
        in_path = os.path.join(RAW_IMAGES, img_name)
        out_path = os.path.join(ROI_OUTPUT, img_name)
        
        # This is a "Safety Check": skip if file already exists from previous run 
        if os.path.exists(out_path): continue
        
        roi_img = processor.process(in_path)
        if roi_img is not None:
            cv2.imwrite(out_path, roi_img)

    final_df = pd.concat(balanced_list)
    final_df.to_csv(FINAL_METADATA, index=False)
    print(f"\nPipeline Complete! Balanced 5-class metadata saved to {FINAL_METADATA}")

if __name__ == "__main__":
    run_targeted_pipeline()