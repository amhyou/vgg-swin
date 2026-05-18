import os
import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import config

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
NUM_CLASSES = config.NUM_CLASSES
TARGET_CLASSES = config.TARGET_CLASSES

TARGET_SIZE = (config.IMG_SIZE, config.IMG_SIZE)
NUM_WORKERS = os.cpu_count() or 4

OUTPUT_DIR = config.RAW_IMAGE_DIR
MASTER_CSV_PATH = config.METADATA_PATH_RAW
OUTPUT_IMG_DIR = OUTPUT_DIR

# Paths to the root downloaded folders
RAW_NIH_DIR = "raw_nih"
RAW_CHEXPERT_DIR = "raw_chexpert/extracted"
# ──────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────

def build_image_index(root_dir):
    """Recursively scans a directory and maps filename -> full absolute path."""
    print(f"Scanning {root_dir} to build a robust file index... (This takes a few seconds)")
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                index[f] = os.path.join(dirpath, f)
    print(f"  -> Found {len(index)} total images in {root_dir}.")
    return index

def find_csv(root_dir, expected_name):
    """Recursively searches for a CSV file in the directory tree."""
    for dirpath, _, filenames in os.walk(root_dir):
        if expected_name in filenames:
            return os.path.join(dirpath, expected_name)
    return None

def find_batch_dirs(base_dir):
    """Finds all CheXpert batch directories inside the extracted folder."""
    batch_dirs = []
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path) and 'CheXpert' in name:
                batch_dirs.append(full_path)
    print(f"  -> Found {len(batch_dirs)} CheXpert batch folder(s): {[os.path.basename(d) for d in batch_dirs]}")
    return batch_dirs

def resolve_chexpert_path(batch_dirs, csv_path_col):
    """
    Strips 'CheXpert-v1.0/train/' prefix from CSV path and searches all batch dirs.
    """
    parts = csv_path_col.replace('\\', '/').split('/')
    relative_path = os.path.join(*parts[2:])
    for batch_dir in batch_dirs:
        candidate = os.path.join(batch_dir, relative_path)
        if os.path.exists(candidate):
            return candidate
    return None

def process_image(task):
    """Resizes a single image to TARGET_SIZE and saves it."""
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    src_path, dst_path = task
    try:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, False
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        ext = os.path.splitext(dst_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            cv2.imwrite(dst_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(dst_path, img_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        return os.path.basename(dst_path), True
    except Exception:
        return None, False

def execute_tasks(tasks, records, dataset_name):
    print(f"Resizing {len(tasks)} images for {dataset_name} using {NUM_WORKERS} CPU cores...")
    successful_ids = set()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_image, tasks), total=len(tasks)))
    for img_id, success in results:
        if success:
            successful_ids.add(img_id)
    final_records = [r for r in records if r['Image_ID'] in successful_ids]
    print(f"Successfully processed {len(final_records)} images.")
    return pd.DataFrame(final_records)

# ─────────────────────────────────────────────────────────
# NIH PROCESSING
# ─────────────────────────────────────────────────────────

def prepare_nih():
    print(f"\n--- Processing NIH Dataset ({NUM_CLASSES}-class) ---")
    if not os.path.exists(RAW_NIH_DIR):
        print(f"Skipping NIH: {RAW_NIH_DIR} not found.")
        return pd.DataFrame()
        
    csv_path = find_csv(RAW_NIH_DIR, "Data_Entry_2017.csv")
    if not csv_path:
        print("Skipping NIH: Data_Entry_2017.csv not found anywhere inside raw_nih/.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    img_index = build_image_index(RAW_NIH_DIR)

    processed_records = []
    tasks = []

    for _, row in df.iterrows():
        labels_str = row['Finding Labels']
        img_id = row['Image Index']

        labels = labels_str.split('|')

        # Map 'No Finding' -> 'Normal'
        if 'No Finding' in labels:
            labels.remove('No Finding')
            labels.append('Normal')

        # Only keep images with at least one of our target classes
        has_target = any(c in TARGET_CLASSES for c in labels)
        if not has_target:
            continue

        record = {
            'Image_ID': f"nih_{img_id}",
            'Dataset': 'NIH',
        }
        for c in TARGET_CLASSES:
            record[c] = 1 if c in labels else 0
            
        processed_records.append(record)

        if img_id in img_index:
            src_path = img_index[img_id]
            dst_path = os.path.join(OUTPUT_IMG_DIR, f"nih_{img_id}")
            tasks.append((src_path, dst_path))

    return execute_tasks(tasks, processed_records, "NIH")

# ─────────────────────────────────────────────────────────
# CHEXPERT PROCESSING
# ─────────────────────────────────────────────────────────

def prepare_chexpert():
    print(f"\n--- Processing CheXpert Dataset ({NUM_CLASSES}-class) ---")
    if not os.path.exists(RAW_CHEXPERT_DIR):
        print(f"Skipping CheXpert: {RAW_CHEXPERT_DIR} not found.")
        return pd.DataFrame()
        
    csv_path = find_csv(RAW_CHEXPERT_DIR, "train.csv")
    if not csv_path:
        print("Skipping CheXpert: train.csv not found anywhere inside raw_chexpert/.")
        return pd.DataFrame()

    batch_dirs = find_batch_dirs(RAW_CHEXPERT_DIR)
    if not batch_dirs:
        print("Skipping CheXpert: Cannot find any 'CheXpert' batch folders after extraction.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df.fillna(0)

    processed_records = []
    tasks = []
    skipped_missing = 0

    for _, row in df.iterrows():
        col_map = {
            'Atelectasis': 'Atelectasis',
            'Cardiomegaly': 'Cardiomegaly',
            'Effusion': 'Pleural Effusion',
            'Normal': 'No Finding',
            'Pneumonia': 'Pneumonia'
        }
        
        drop = False
        has_target = False
        record = {'Dataset': 'CheXpert'}
        
        for cls in TARGET_CLASSES:
            chex_col = col_map[cls]
            val = row.get(chex_col, 0)
            if val == -1:
                drop = True
                break
            final_val = 1 if val == 1 else 0
            record[cls] = final_val
            if final_val == 1:
                has_target = True

        if drop or not has_target:
            continue
            
        # Only keep Frontal views (AP/PA), ignore Lateral — case-insensitive!
        if 'frontal' not in str(row['Path']).lower():
            continue

        # Multi-batch path resolution
        src_path = resolve_chexpert_path(batch_dirs, row['Path'])
        if src_path is None:
            skipped_missing += 1
            continue

        safe_filename = 'chexpert_' + row['Path'].replace('CheXpert-v1.0/', '').replace('/', '_')
        record['Image_ID'] = safe_filename
        
        processed_records.append(record)
        dst_path = os.path.join(OUTPUT_IMG_DIR, safe_filename)
        tasks.append((src_path, dst_path))

    if skipped_missing > 0:
        print(f"  [INFO] Skipped {skipped_missing} images not found on disk.")

    return execute_tasks(tasks, processed_records, "CheXpert")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MASTER_CSV_PATH), exist_ok=True)

    # CheXpert first for quick verification
    chexpert_df = prepare_chexpert()
    nih_df = prepare_nih()

    if chexpert_df.empty and nih_df.empty:
        print("\nNo data processed. Ensure raw data folders exist and are populated.")
        return

    master_df = pd.concat([nih_df, chexpert_df], ignore_index=True)

    master_df.to_csv(MASTER_CSV_PATH, index=False)

    print("\n==================================================")
    print(f"✅ {NUM_CLASSES}-Class Data Preparation Complete!")
    print(f"Total Images: {len(master_df)}")
    print(f"Class Counts:")
    for cls in TARGET_CLASSES:
        print(f" - {cls}: {master_df[cls].sum()}")
    print(f"Master CSV saved to: {MASTER_CSV_PATH}")
    print("==================================================")
    print(f"\nNext step: zip -r {OUTPUT_DIR}.zip {OUTPUT_DIR}/ metadata/")

if __name__ == "__main__":
    main()
