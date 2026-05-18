import os

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
# Priority order: ['Effusion', 'Normal','Cardiomegaly', 'Atelectasis', 'Pneumonia']
# Example: NUM_CLASSES = 4 gives ['Effusion', 'Normal', 'Cardiomegaly', 'Atelectasis']
NUM_CLASSES = 4
MASTER_CLASSES = ['Effusion', 'Normal','Cardiomegaly', 'Atelectasis', 'Pneumonia']
TARGET_CLASSES = MASTER_CLASSES[:NUM_CLASSES]

IMG_SIZE = 224

# ─── DATASET MODE ─────────────────────────────────────────
SINGLE_LABEL_MODE = True
SAMPLES_PER_CLASS = 10_000

# ─── PATHS ────────────────────────────────────────────────
# Standardized zip extraction paths
RAW_IMAGE_DIR = f"raw_{IMG_SIZE}"
ROI_IMAGE_DIR = f"roi_{IMG_SIZE}"

# Metadata is extracted outside the image folders
METADATA_PATH_RAW = os.path.join("metadata", "DATA_RAW.csv")
METADATA_PATH_ROI = os.path.join("metadata", "DATA_ROI.csv")

# Active metadata (used by C_train, D_test, F_gradcam)
METADATA_PATH = METADATA_PATH_ROI

# Model checkpoints
CHECKPOINT_DIR = "weights"

# Results / logs
RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.csv")

# ─── TRAINING HYPERPARAMETERS (Optimized via Optuna) ──────
BATCH_SIZE = 128
ACCUMULATION_STEPS = 1          # Effective batch = 64
EARLY_STOP_PATIENCE = 8
MAX_GRAD_NORM = 1.0

# Best Optuna parameters
LR = 2.73970591357564e-05
WEIGHT_DECAY = 0.00018567760436897365
MIXUP_ALPHA = 0.3215717961727873
DROP_PATH_RATE = 0.15694918750476322
OPTIMIZER = "Adam"

# Hardcoded optimal architectural parameters
LABEL_SMOOTHING = 0.1
HEAD_DROPOUT = 0.3
