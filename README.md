# Chest X-Ray Classification with Hybrid VGG-Swin Transformer

This project implements a comprehensive chest X-ray classification system for detecting lung diseases including Atelectasis, Cardiomegaly, Effusion, Pneumonia, and Normal cases. The system employs a hybrid architecture combining VGG16 and Swin Transformer to leverage the strengths of both models: local feature detection from convolutional networks and global context understanding from vision transformers.

## Overview

The project consists of several key components:
- **Lung Segmentation**: U-Net model for isolating lung regions from chest X-rays
- **Classification Model**: Hybrid architecture combining VGG16 backbone with Swin Transformer
- **Training Pipeline**: Complete training and evaluation workflow
- **Visualization**: Grad-CAM analysis for model interpretability

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)


## Usage with Jupyter Notebook (Recommended) 

For an interactive and user-friendly experience, use the provided Jupyter notebook that guides you through all steps:

```bash
# Open the master notebook
jupyter notebook notebooks/master.ipynb
```

The notebook includes all the same steps as the command-line approach below, but in an interactive format perfect for experimentation.

## Command-Line Usage

If you prefer using the command line or need to automate the process, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/amhyou/vgg-swin.git
cd vgg-swin
```

2. **Install dependencies:**

If you are in a local environment, preferably to install the exaustive list of dependencies
```bash
pip install -r requirements.txt
```
If you are in a cloud environment (Colab or Kaggle), you can simply execute this
```bash
pip install -q gdown
```

## Dataset Preparation

### Option 1: Preprocessed Dataset (Recommended for Training)
Download the preprocessed dataset containing isolated lung regions:

```bash
# Download and extract the preprocessed dataset
gdown 1M1n98Bq1PRgpiuI0Vk02LAO0sLkEWYuy -O roi.zip
unzip roi.zip
```

This will create a `data/roi/` directory with pre-segmented lung images.

### Option 2: Raw Dataset (Required for Grad-CAM Visualization)
For complete analysis including Grad-CAM visualization:

```bash
# Download and extract raw dataset
gdown 1q3kj83U1A667GouyIJJkcWVrkSommIq -O raw.zip
unzip raw.zip
```

This creates a `data/raw/` directory with original chest X-ray images.

## Project Structure

```
thesis/
├── models/
│   ├── main.py         # VGGSwinHybridNet architecture
│   └── unet.py         # U-Net implementation
├── A_preprocess.py     # Data preprocessing script
├── B_train.py          # Model training script
├── C_test.py           # Model evaluation script
├── D_visualize.py      # Results visualization
├── E_gradcam.py        # Grad-CAM analysis
├── requirements.txt    # Python dependencies
├── metadata/           # Dataset metadata
├── results/            # Output results and logs
└── weights/            # Model checkpoints
|
└── data/             ## Downloaded via earler 'gdown' commands
│   ├── raw/          # Raw dataset
│   └── roi/          # ROI Isolated dataset
```

### Step 1: Data Preprocessing (Optional)
If you have raw data and want to perform custom preprocessing:

```bash
python A_preprocess.py
```

### Step 2: Model Training
Train the hybrid VGG-Swin Transformer model:

```bash
python B_train.py
```

**What this does:**
- Loads preprocessed chest X-ray data
- Trains the classification model for 5 classes
- Saves model checkpoints to `weights/` directory
- Logs training metrics to `results/comprehensive_training_log.csv`

**Expected output:**
- Trained model weights saved as `weights/best_model.pth`
- Training logs and metrics in CSV format

### Step 3: Model Testing
Evaluate the trained model on test data:

```bash
python C_test.py
```

**What this does:**
- Loads the best trained model
- Evaluates performance on test set
- Generates comprehensive test report

**Expected output:**
- Test results saved to `results/final_test_report.csv`
- Performance metrics (accuracy, F1-score, precision, recall, AUC)

### Step 4: Results Visualization
Create visualizations of training results:

```bash
python D_visualize.py
```

**What this does:**
- Generates plots for training curves, confusion matrices, and metrics
- Creates comparative visualizations of model performance

**Expected output:**
- Various plots saved to `results/` directory

### Step 5: Grad-CAM Analysis (Requires Raw Dataset)
Perform interpretability analysis using Grad-CAM:

```bash
python E_gradcam.py
```

**What this does:**
- Applies Grad-CAM to visualize model attention on raw X-ray images
- Generates heatmaps showing which regions influenced predictions

**Expected output:**
- Grad-CAM visualization images saved to `results/` directory

## Model Architecture

### Lung Segmentation (U-Net)
- **Input**: Grayscale chest X-ray images (224x224)
- **Architecture**: Classic U-Net with skip connections
- **Output**: Binary lung segmentation masks
- **Purpose**: Isolates lung regions for focused classification

### Classification Model (VGGSwinHybridNet)
- **Backbone**: Truncated VGG16 (first 16 layers)
- **Bridge**: 1x1 convolution adapting channels (256→96)
- **Head**: Swin Transformer Tiny
- **Output**: 5-class classification (Atelectasis, Cardiomegaly, Effusion, Pneumonia, Normal)

## Configuration

Key parameters can be modified in the respective Python files:

- **Batch size**: Modify `BATCH_SIZE` in `B_train.py`
- **Image size**: Change `img_size` in `models/processor.py`
- **Target classes**: Update `TARGET_CLASSES` in `B_train.py`
- **Training epochs**: Adjust in the training loop

## Results and Outputs

All results are saved to the `results/` directory:

- `comprehensive_training_log.csv`: Training metrics per epoch
- `final_test_report.csv`: Test set evaluation results
- Various plots and visualizations
- Grad-CAM heatmaps (if raw dataset is used)

### Downloading Results

To download all results:

```bash
zip -r results.zip results/
```