import os
import random
import numpy as np

# ======================================
# Base Directory
# ======================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ======================================
# Dataset
# ======================================

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

DATASET_PATH = os.path.join(
    DATASET_DIR,
    "parkinsons.csv"
)

# ======================================
# Output Directories
# ======================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

GRAPH_DIR = os.path.join(
    OUTPUT_DIR,
    "graphs"
)

LOG_DIR = os.path.join(
    OUTPUT_DIR,
    "logs"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

DATABASE_PATH = os.path.join(
    OUTPUT_DIR,
    "parkinsons.db"
)

# ======================================
# Model Files
# ======================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

# ======================================
# Target & Feature Settings
# ======================================

TARGET_COLUMN = "status"

DROP_COLUMNS = [
    "name"
]

# ======================================
# Training Configuration
# ======================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

CV_FOLDS = 5

# ======================================
# Random Seed
# ======================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

# ======================================
# Create Directories
# ======================================

directories = [

    DATASET_DIR,

    OUTPUT_DIR,

    MODEL_DIR,

    GRAPH_DIR,

    LOG_DIR,

    REPORT_DIR

]

for directory in directories:

    os.makedirs(

        directory,

        exist_ok=True

    )

# ======================================
# Configuration Summary
# ======================================

print("=" * 60)

print("Parkinson's Disease Prediction Configuration")

print("=" * 60)

print("Dataset :", DATASET_PATH)

print("Target :", TARGET_COLUMN)

print("Test Size :", TEST_SIZE)

print("Cross Validation :", CV_FOLDS)

print("Random State :", RANDOM_STATE)

print("=" * 60)