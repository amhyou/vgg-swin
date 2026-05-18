import os

# Must be set BEFORE TensorFlow is imported
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning logs

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor

# Enable GPU memory growth to avoid OOM and cuDNN allocation issues
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU detected: {[g.name for g in gpus]} — memory growth enabled.")
else:
    print("No GPU detected. Running on CPU.")

from unet import get_unet

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
import config
NUM_CLASSES = config.NUM_CLASSES
TARGET_CLASSES = config.TARGET_CLASSES
BATCH_SIZE = 32   # Set to 4 for local testing if GPU memory is low, 32-64 for Vast.ai

# In Kaggle working dir fallback
OUTPUT_BASE = "/kaggle/working" if os.path.exists('/kaggle/working') else "."

# Input Paths
# We assume the user has the folder structure standardized in config.py
INPUT_CSV = os.path.join(OUTPUT_BASE, config.METADATA_PATH_RAW)
RAW_IMAGES = os.path.join(OUTPUT_BASE, config.RAW_IMAGE_DIR)

# Output Paths
ROI_OUTPUT = os.path.join(OUTPUT_BASE, config.ROI_IMAGE_DIR)
FINAL_METADATA = os.path.join(OUTPUT_BASE, config.METADATA_PATH_ROI)

WEIGHTS = "weights/cxr_reg_weights.best.hdf5"
IMG_SIZE = config.IMG_SIZE
# ──────────────────────────────────────────────────────────

def load_and_preprocess_single_image(args):
    """Loads a single image, applies CLAHE, and prepares it for the model."""
    img_name, in_path, img_size = args
    
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return img_name, None, None
        
    if img.shape[:2] != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    img_in = enhanced.astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, axis=-1) 
    
    return img_name, enhanced, img_in

def save_single_roi(args):
    """Applies mask and saves the ROI image to disk."""
    out_path, enhanced, mask_binary = args
    roi_img = cv2.bitwise_and(enhanced, enhanced, mask=mask_binary)
    cv2.imwrite(out_path, roi_img)

class BatchedLungProcessor:
    def __init__(self, weights_path, img_size=384, batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = get_unet((img_size, img_size, 1))

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at {weights_path}. Please download them first.")
        self.model.load_weights(weights_path)

    def process_dataset(self, image_names, raw_dir, out_dir):
        """Processes a list of image names in batches using ThreadPool for I/O."""
        
        # Filter out images that already exist in output
        pending_images = [img for img in image_names if not os.path.exists(os.path.join(out_dir, img))]
        if not pending_images:
            return

        print(f"Processing {len(pending_images)} images in batches of {self.batch_size}...")

        # Create batches
        batches = [pending_images[i:i + self.batch_size] for i in range(0, len(pending_images), self.batch_size)]
        
        # We use a ThreadPoolExecutor to speed up loading and saving (I/O bound)
        # while the main thread handles GPU prediction (Compute bound)
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            for batch_names in tqdm(batches):
                
                # 1. Parallel Load & Preprocess (CPU)
                load_args = [(name, os.path.join(raw_dir, name), self.img_size) for name in batch_names]
                loaded_results = list(executor.map(load_and_preprocess_single_image, load_args))
                
                valid_names = []
                enhanced_imgs = []
                model_inputs = []
                
                for name, enhanced, img_in in loaded_results:
                    if enhanced is not None:
                        valid_names.append(name)
                        enhanced_imgs.append(enhanced)
                        model_inputs.append(img_in)
                
                if not model_inputs:
                    continue
                    
                # 2. Batched Prediction (GPU)
                batch_tensor = np.array(model_inputs)
                masks = self.model.predict(batch_tensor, verbose=0)
                masks_binary = (masks > 0.5).astype(np.uint8)
                
                # 3. Parallel Masking & Save (CPU)
                save_args = [
                    (os.path.join(out_dir, name), enhanced_imgs[i], masks_binary[i, :, :, 0])
                    for i, name in enumerate(valid_names)
                ]
                # Execute saves in parallel
                list(executor.map(save_single_roi, save_args))


def download_weights_if_needed():
    if not os.path.exists(WEIGHTS):
        print("U-Net weights not found locally. Attempting to download via Kaggle API...")
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        os.system("kaggle kernels output nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset -p ./weights/")
        
        files = os.listdir("./weights/")
        hdf5_files = [f for f in files if f.endswith('.hdf5') or f.endswith('.h5')]
        if hdf5_files and hdf5_files[0] != os.path.basename(WEIGHTS):
            os.rename(os.path.join("./weights", hdf5_files[0]), WEIGHTS)
            
        if os.path.exists(WEIGHTS):
            print("Successfully downloaded U-Net weights!")
        else:
            print(f"Warning: Could not download or find {WEIGHTS}")

def run_targeted_pipeline(args):
    if args.download_weights:
        download_weights_if_needed()

    if not os.path.exists(ROI_OUTPUT): os.makedirs(ROI_OUTPUT, exist_ok=True)
    if not os.path.exists(os.path.dirname(FINAL_METADATA)): os.makedirs(os.path.dirname(FINAL_METADATA), exist_ok=True)

    print(f"Step 1: Loading {NUM_CLASSES}-class metadata...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please run A_prepare_data.py first.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    target_df = df[df[TARGET_CLASSES].sum(axis=1) > 0].copy()

    print("\nStep 2: Calculating dynamic class balancing...")
    class_counts = {cls: target_df[cls].sum() for cls in TARGET_CLASSES}
    target_samples_per_class = max(class_counts.values())
    
    print(f"Class counts: {class_counts}")
    print(f"Targeting {target_samples_per_class} samples per class (balancing to maximum).")

    balanced_list = []
    unique_images_to_process = set()

    for cls in TARGET_CLASSES:
        subset = target_df[target_df[cls] == 1]
        count = len(subset)
        
        if count == 0:
            continue
            
        multiplier = max(1, target_samples_per_class // count)
        print(f" - {cls}: {count} base images. Augmentation factor: {multiplier}x")
        
        for i in range(multiplier):
            temp = subset.copy()
            temp['aug_instance'] = i 
            balanced_list.append(temp)
            unique_images_to_process.update(subset['Image_ID'].tolist())

    print(f"\nStep 3: Batched ROI Isolation for {len(unique_images_to_process)} unique images...")
    try:
        processor = BatchedLungProcessor(WEIGHTS, img_size=IMG_SIZE, batch_size=args.batch_size)
        processor.process_dataset(list(unique_images_to_process), RAW_IMAGES, ROI_OUTPUT)
    except FileNotFoundError as e:
        print(e)
        print("Run with --download-weights to fetch them automatically.")
        return

    final_df = pd.concat(balanced_list)
    final_df.to_csv(FINAL_METADATA, index=False)
    print(f"\nPipeline Complete! Balanced {NUM_CLASSES}-class metadata saved to {FINAL_METADATA}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Chest X-Rays with U-Net")
    parser.add_argument('--download-weights', action='store_true', help="Download U-Net weights from Kaggle")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size for GPU inference")
    args = parser.parse_args()
    
    run_targeted_pipeline(args)