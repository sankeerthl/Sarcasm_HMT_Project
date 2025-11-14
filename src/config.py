# ====================================================
# src/config.py — Fully Updated SARC-MT-CLIP++ Unified Configuration
# ====================================================
"""
Central configuration for SARC-MT-CLIP++.

This file gathers training / model / data / inference constants so other modules
can import cleanly. Tweak values here for experiments and to ensure reproducibility.
"""

import os
import torch
import random
import numpy as np

# ====================================================
# Silence Hugging Face Windows symlink warnings
# ====================================================
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Safe import of CLIPProcessor (may fail in some trimmed envs)
try:
    from transformers import CLIPProcessor  # type: ignore
except Exception:
    CLIPProcessor = None  # type: ignore


# ====================================================
# 🚀 Device & Seed
# ====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Set Python / NumPy / Torch seeds for reproducibility and deterministic dataloaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # deterministic behaviour (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ====================================================
# 📂 Paths
# ====================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_ROOT, "raw_images")
TEXT_MASKED_IMAGE_DIR = os.path.join(DATA_ROOT, "text_masked_images")
LABEL_FILE = os.path.join(DATA_ROOT, "labels.csv")

# output / checkpoints / artifacts
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ====================================================
# 🧩 Data Splits
# ====================================================
N_FOLDS = 5
GROUP_BY_TEMPLATE = True               # group split by template_id to avoid template leakage
STRATIFY_BY = "sarcasm_intensity"      # column name used for stratification
VAL_SPLIT = 0.1
RANDOM_STATE_SPLIT = RANDOM_SEED


# ====================================================
# 🧠 CLIP Encoder Configuration
# ====================================================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_TEXT_FEATURE_DIM = 512
CLIP_IMAGE_FEATURE_DIM = 512
FREEZE_CLIP = True
UNFREEZE_LAST_BLOCKS = 2
UNFREEZE_EPOCH = 15

if CLIPProcessor is not None:
    try:
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    except Exception as e:
        print(f"⚠️ Warning: CLIPProcessor load failed ({e}). Continuing without processor.")
        CLIP_PROCESSOR = None
else:
    CLIP_PROCESSOR = None


# ====================================================
# 🔗 Fusion Transformer
# ====================================================
FUSION_TYPE = "cross_modal_transformer"
FUSION_LAYERS = 2
FUSION_HEADS = 8
FUSION_HIDDEN_DIM = 768
FUSION_DROPOUT = 0.5
FUSION_NORM = True


# ====================================================
# 🔽 Classifier Dropout
# ====================================================
CLASSIFIER_DROPOUT = 0.5


# ====================================================
# 🎯 Multi-Task Output Heads
# ====================================================
TASKS = {
    "sarcasm_presence": {"type": "binary", "loss": "bce_focal"},
    "sarcasm_intensity": {"type": "ordinal", "loss": "coral"},
    "humor": {"type": "multiclass", "num_classes": 3, "loss": "ce"},
    "offense": {"type": "multiclass", "num_classes": 3, "loss": "ce"},
    "sentiment": {"type": "multiclass", "num_classes": 3, "loss": "ce"},
    "motivation": {"type": "multiclass", "num_classes": 4, "loss": "ce"},
}

NUM_CLASSES = 3  # sarcasm intensity (0, 1, 2)

LOSS_WEIGHTS = {
    "sarcasm_presence": 2.0,
    "sarcasm_intensity": 2.0,
    "humor": 1.0,
    "offense": 1.0,
    "sentiment": 1.0,
    "motivation": 0.5,
}

USE_DYNAMIC_UNCERTAINTY = True


# ====================================================
# 🧩 Auxiliary Losses
# ====================================================
CONTRASTIVE_TEMP = 0.07
INCONGRUITY_MARGIN = 0.1
CONSISTENCY_WEIGHT_AUX = 0.3         # weight for auxiliary head consistency
CONSISTENCY_WEIGHT_SARCASM = 0.3     # weight for presence/intensity consistency


# ====================================================
# ⚖️ Class Imbalance / Focal Loss
# ====================================================
USE_CB_FOCAL = True
FOCAL_GAMMA = 2.0
OVERSAMPLE_INTENSITY = True
OVERSAMPLE_MAX_RATIO = 2.0


def compute_class_weights(labels, num_classes):
    """
    Compute inverse-frequency normalized class weights as a torch tensor on DEVICE.
    labels: iterable of ints
    num_classes: int
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ====================================================
# 🧮 Training Hyperparameters
# ====================================================
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 20
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
USE_AMP = True  # automatic mixed precision

LR_HEADS = 2e-4
LR_FUSION = 1e-4
LR_ENCODERS = 1e-5
OPTIMIZER = "adamw"

SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05
PATIENCE = 4


# ====================================================
# 🧰 Regularization & Augmentation
# ====================================================
IMAGE_AUGMENTATIONS = {
    "random_resized_crop": (0.9, 1.0),
    "horizontal_flip": True,
    "color_jitter": True,
    "gaussian_noise": True,
}
TEXT_AUGMENT = False


# ====================================================
# 📊 Calibration & Evaluation
# ====================================================
USE_TEMPERATURE_SCALING = True
CALIBRATION_SPLIT = "val"
TUNE_THRESHOLDS = True
PRIMARY_METRIC = "macro_f1_intensity"
SECONDARY_METRICS = ["presence_f1", "ordinal_mae"]


# ====================================================
# 🧪 Inference & Ensemble
# ====================================================
TTA = True
TTA_CROPS = 5
TTA_FLIP = True
ENSEMBLE_MODELS = 3
USE_SWA = True


# ====================================================
# 💾 Saving & Logging
# ====================================================
LOG_INTERVAL = 50
SAVE_BEST_MODEL = True
SAVE_CHECKPOINTS = True
VERBOSE = True


# ====================================================
# 🔄 DataLoader Reproducibility
# ====================================================
def worker_init_fn(worker_id):
    """Worker init for deterministic behavior in multi-worker DataLoader."""
    seed = RANDOM_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


WORKERS = min(8, max(0, (os.cpu_count() or 4) - 1))


# ====================================================
# 🧠 Veracity / NLI Module
# ====================================================
NLI_MODEL_NAME = "roberta-large-mnli"
VERACITY_OUTPUT_DIM = 3


# ====================================================
# 🧾 Misc / Experiment metadata
# ====================================================
EXPERIMENT_NAME = "SARC-MT-CLIP++"
NOTES = "Research-grade config — track changes and hyperparams in lab notebook."


# ====================================================
# 🧩 Exportable Symbols
# ====================================================
__all__ = [
    "DEVICE", "RANDOM_SEED", "set_seed",
    "PROJECT_ROOT", "DATA_ROOT", "IMAGE_DIR", "TEXT_MASKED_IMAGE_DIR", "LABEL_FILE",
    "N_FOLDS", "GROUP_BY_TEMPLATE", "STRATIFY_BY",
    "CLIP_MODEL_NAME", "CLIP_PROCESSOR",
    "FUSION_HIDDEN_DIM", "FUSION_DROPOUT", "CLASSIFIER_DROPOUT",
    "LOSS_WEIGHTS", "USE_DYNAMIC_UNCERTAINTY", "CONTRASTIVE_TEMP",
    "INCONGRUITY_MARGIN", "OVERSAMPLE_INTENSITY", "WORKERS",
    "BATCH_SIZE", "NUM_EPOCHS", "LR_HEADS", "LR_FUSION", "LR_ENCODERS",
    "USE_TEMPERATURE_SCALING", "OUTPUT_DIR", "CHECKPOINT_DIR", "METRICS_DIR",
    "NLI_MODEL_NAME", "VERACITY_OUTPUT_DIM", "NUM_CLASSES", "USE_AMP",
    "FOCAL_GAMMA", "PATIENCE", "USE_CB_FOCAL",
    "WEIGHT_DECAY", "GRAD_CLIP", "USE_SWA", "LOSS_WEIGHTS", "LABEL_SMOOTHING",
    "CONSISTENCY_WEIGHT_AUX", "CONSISTENCY_WEIGHT_SARCASM", "IMAGE_AUGMENTATIONS",
    "EXPERIMENT_NAME", "NOTES", "CLIP_TEXT_FEATURE_DIM", "CLIP_IMAGE_FEATURE_DIM"
]
