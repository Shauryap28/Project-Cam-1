"""
Central configuration for the camouflaged-soldier detection pipeline.

Everything that you used to edit inside individual Colab cells lives here so you
never have to hunt through code to change a path or a hyperparameter again.
Override any of these with environment variables if you want (see os.getenv).
"""
import os
import torch

# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ----------------------------------------------------------------------------
# Paths  (edit these for your machine / Drive)
# ----------------------------------------------------------------------------
# Stage 1 — ZoomNeXt detector
ZOOMNEXT_DIR = os.getenv("ZOOMNEXT_DIR", "/content/drive/MyDrive/ZoomNeXt_folder/ZoomNeXt")
ZOOMNEXT_WEIGHTS = os.getenv(
    "ZOOMNEXT_WEIGHTS",
    "/content/drive/MyDrive/ZoomNeXt_folder/ZoomNeXt/pretrained_weights/pvtv2-b4-zoomnext.pth",
)

# Datasets
SPLIT_DATASET_ROOT = os.getenv("SPLIT_DATASET_ROOT", "/content/drive/MyDrive/split_dataset")
CROPPED_ROOT = os.getenv("CROPPED_ROOT", "/content/drive/MyDrive/croppedcamdata")

# Classifier checkpoints
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/content/drive/MyDrive/swin_checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "swin_v2_small_body.pth")

# ----------------------------------------------------------------------------
# Detection (Stage 1) settings
# ----------------------------------------------------------------------------
DET_BASE_H, DET_BASE_W = 384, 384       # ZoomNeXt multi-scale base resolution
DET_THRESHOLD = 0.5                      # sigmoid threshold for the mask
DET_MIN_COVERAGE = 0.005                 # min mask area (fraction) to count as a detection
BBOX_PADDING = 0.15                      # padding around the detected box

# ----------------------------------------------------------------------------
# Classifier (Stage 2) settings
# ----------------------------------------------------------------------------
CLASSIFIER_NAME = "swinv2_small_window8_256"
NUM_CLASSES = 2
INPUT_SIZE = 256
CROP_SIZE = 224                          # size cropped soldiers are saved at (Stage-1 output)
CLASS_NAMES = ["cam", "no_cam"]          # index 0 = cam, 1 = no_cam

# Training
BATCH_SIZE = 24
PHASE1_EPOCHS = 8
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 20
PHASE2_LR = 5e-5
WEIGHT_DECAY = 0.05
PATIENCE = 5
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

# ImageNet normalisation
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------------
# Inference fusion + confidence thresholds
# ----------------------------------------------------------------------------
FUSION_MIN_COVERAGE = 5.0                # % coverage below which we trust the full image only
FUSION_MAX_CROP_WEIGHT = 0.7
HIGH_CONF = 75.0
LOW_CONF = 65.0
