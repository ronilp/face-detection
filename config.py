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
ALLOWED_CLASSES = []
NUM_CLASSES = len(ALLOWED_CLASSES)

### Miscellaneous Config
MODEL_PREFIX = "model_name"
BATCH_SIZE = 16
RANDOM_SEED = 629

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
GPU_MODE = torch.cuda.is_available()