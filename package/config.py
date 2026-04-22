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

# Disease tag for the active pipeline run.
# 'adni' = existing ADNI dataset (ImageFolder).
# Other values (e.g. 'chestxray14', 'isic2024', 'retinamnist') select a
# disease registered in the BDA registry.
DISEASE_ID = "adni"

# Dataset layout used when building the registry for the first time.
# One of: 'imagefolder', 'chestxray14', 'isic2024', 'medmnist_png', 'medmnist_hf'
DATASET_LAYOUT = "imagefolder"

# Where BDA stores its Parquet registry files (one per disease)
REGISTRY_DIR = "registry/"

# Where BDA stores pre-processed WebDataset .tar shards
SHARDS_DIR = "shards/"

# Whether to use the BDA data pipeline.
# True  → use DiseaseRegistry + WebDataset shards (or RegistryDataset fallback)
# False → legacy FullDataset / ImageFolder (original behaviour, ADNI only)
USE_BDA_PIPELINE = True

# Whether to build the registry + shards automatically on first run if they
# do not exist yet.  Set False if you want to manage ETL separately.
AUTO_RUN_PIPELINE = True

# Optional CSV annotation path (required for 'chestxray14' and 'isic2024' layouts)
DATASET_CSV = None  # e.g. "staging/chestxray14/Data_Entry_2017.csv"

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

# ── MLflow tracking ───────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "mlruns/"
USE_MLFLOW = True  # Set False to use CSV-only logging (legacy behaviour)

# Create necessary directories
DATA_DIR.mkdir(exist_ok = True)
LOG_DIR.mkdir(exist_ok = True)
SAVE_DIR.mkdir(exist_ok = True)