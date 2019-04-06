# File: config.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 2:04 AM

import torch

### Learning Parameters
BASE_LR = 1e-4
TRAIN_EPOCHS = 50
EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_PATIENCE = 10

### Dataset Config
DATA_DIR = "data"
ALLOWED_CLASSES = ["face", "no_face"]
NUM_CLASSES = len(ALLOWED_CLASSES)
num_workers = 4
dataset_size = 60000
train_ratio = 0.8


### Miscellaneous Config
MODEL_PREFIX = "model_name"
RANDOM_SEED = 629
BATCH_SIZE = 64

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")