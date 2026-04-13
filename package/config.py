import os
import torch
from pathlib import Path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
ROOT               = Path.cwd()
DATA_DIR           = ROOT       / "data"
OUTPUT_DIR         = ROOT       / "output"
LOG_DIR            = OUTPUT_DIR / "logs"
SAVE_DIR           = OUTPUT_DIR / "saved_models"
TEMP_WEIGHTS_PATH  = SAVE_DIR   / "temp_model_weights.pth"
WEIGHTS_PATH       = SAVE_DIR   / "swin_model_weights.pth"
CLASS_NAMES_PATH   = SAVE_DIR   / "swin_class_names.txt"
CLASS_WEIGHTS_PATH = SAVE_DIR   / "class_weights.npy"

# Model specific
MODEL_NAME = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
IMG_SIZE   = 224

NUM_CLASSES = 4

# Classifier Configuration
OPTIMIZE_METRIC  = 'recall'  # Primary metric: 'recall', 'accuracy', 'f1', 'precision'
MIN_DELTA_METRIC = 0.001  # Minimum improvement threshold for early stopping
MIN_DELTA        = 0.001  # For legacy compatibility

# Training hyperparameters
EPOCHS      = 100  # Total epochs (will be distributed in progressive training)
NFOLDS      = 5
BATCH_SIZE  = 32
NUM_WORKERS = 4
PRETRAINED  = True
TEST_SPLIT  = 0.2
PATIENCE    = 20
LR          = 1e-4

# Optimization settings
USE_AMP            = True  # Automatic Mixed Precision
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

# Memory management
EMPTY_CACHE_FREQUENCY = 1
SAVE_BEST_ONLY        = True

# Create necessary directories
DATA_DIR.mkdir(exist_ok = True)
LOG_DIR.mkdir(exist_ok = True)
SAVE_DIR.mkdir(exist_ok = True)