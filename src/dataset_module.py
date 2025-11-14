# ====================================================
# src/dataset_module.py — Fully Updated SARC-MT-CLIP++ Dataset & Loader (Research-Grade)
# ====================================================
"""
Implements multimodal dataset handling for memes with:
  • Dual image views (original + text-masked)
  • Template-grouped stratified k-folds (leak-proof)
  • Class-balanced oversampling
  • Clean CLIP-based tokenization and augmentation

Supports reproducible data splits, augmentation controls, and auxiliary tasks.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    CLIP_PROCESSOR,
    IMAGE_DIR,
    TEXT_MASKED_IMAGE_DIR,
    RANDOM_SEED,
    worker_init_fn,
    IMAGE_AUGMENTATIONS,
    STRATIFY_BY,
    OVERSAMPLE_INTENSITY,
    OVERSAMPLE_MAX_RATIO,
    IMAGE_SIZE,
)


# ====================================================
# 🧩 Utility: deterministic shuffling
# ====================================================
def _seed_worker(worker_id):
    """Ensures deterministic random behavior across dataloader workers."""
    seed = RANDOM_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


# ====================================================
# 🚀 MemeDataset — dual-view multimodal CLIP input
# ====================================================
class MemeDataset(Dataset):
    """
    Dataset yielding:
      • Original and text-masked image views
      • Tokenized meme text (transcribed/cleaned)
      • Multi-task label dictionary
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str = IMAGE_DIR,
        masked_image_dir: str = TEXT_MASKED_IMAGE_DIR,
        processor=CLIP_PROCESSOR,
        is_train: bool = True,
        max_len: int = 77,
        augment_cfg: dict = IMAGE_AUGMENTATIONS,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.masked_image_dir = masked_image_dir
        self.processor = processor
        self.is_train = is_train
        self.max_len = max_len
        self.augment_cfg = augment_cfg
        self.task_columns = [c for c in self.df.columns if c.startswith("label_")]
        self.has_template = "template_id" in self.df.columns

        # === Image transforms ===
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        train_tfms = [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=self.augment_cfg.get("random_resized_crop", (0.9, 1.0)),
            ),
        ]
        if self.augment_cfg.get("horizontal_flip", True):
            train_tfms.append(transforms.RandomHorizontalFlip())
        if self.augment_cfg.get("color_jitter", True):
            train_tfms.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            )
        train_tfms.append(transforms.ToTensor())
        train_tfms.append(normalize)

        self.image_transform_train = transforms.Compose(train_tfms)
        self.image_transform_val = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Safe image loading ---
        def safe_open(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                # fallback blank white image if missing/corrupted
                return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))

        img_path = os.path.join(self.image_dir, row["image_name"])
        masked_path = os.path.join(self.masked_image_dir, row["image_name"])
        img = safe_open(img_path)
        img_masked = safe_open(masked_path)

        transform = self.image_transform_train if self.is_train else self.image_transform_val
        img = transform(img)
        img_masked = transform(img_masked)

        # --- Text tokenization ---
        text = str(row.get("text_corrected", "")).strip()
        if self.processor is not None:
            text_inputs = self.processor(
                text=[text],
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )
            input_ids = text_inputs["input_ids"].squeeze(0)
            attention_mask = text_inputs["attention_mask"].squeeze(0)
        else:
            # fallback if processor missing
            input_ids = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        # --- Multi-task labels ---
        labels = {
            col: torch.tensor(row[col], dtype=torch.long)
            for col in self.task_columns
        }

        sample = {
            "image_orig": img,
            "image_masked": img_masked,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text_raw": text,
        }
        if self.has_template:
            sample["template_id"] = row["template_id"]

        return sample


# ====================================================
# ⚙️ Collate Function — Safe Batching
# ====================================================
def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    collated = {}
    collated["image_orig"] = torch.stack([b["image_orig"] for b in batch])
    collated["image_masked"] = torch.stack([b["image_masked"] for b in batch])

    collated["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    collated["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )

    first_labels = batch[0]["labels"]
    collated["labels"] = {
        k: torch.stack([b["labels"][k] for b in batch]) for k in first_labels
    }

    collated["text_raw"] = [b["text_raw"] for b in batch]
    if "template_id" in batch[0]:
        collated["template_id"] = [b["template_id"] for b in batch]

    return collated


# ====================================================
# ⚖️ Oversampling by Sarcasm Intensity
# ====================================================
def oversample_by_intensity(df, label_col="label_sarcasm_intensity"):
    """Oversample minority intensity classes up to a capped ratio."""
    counts = df[label_col].value_counts()
    max_count = counts.max()
    dfs = []
    for cls, count in counts.items():
        target_n = int(min(max_count * OVERSAMPLE_MAX_RATIO, max_count * 2))
        subset = df[df[label_col] == cls]
        if count < target_n:
            add_df = subset.sample(n=target_n, replace=True, random_state=RANDOM_SEED)
        else:
            add_df = subset
        dfs.append(add_df)
    df_aug = pd.concat(dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_aug


# ====================================================
# 📦 DataLoader Builder
# ====================================================
def create_data_loader(
    dataframe,
    batch_size=16,
    num_workers=2,
    is_train=True,
    use_oversampling=True,
):
    """
    Builds torch DataLoader with safe reproducibility and optional oversampling.
    """
    df = dataframe.copy().reset_index(drop=True)
    label_col = f"label_{STRATIFY_BY}"
    if is_train and OVERSAMPLE_INTENSITY and use_oversampling and label_col in df.columns:
        df = oversample_by_intensity(df, label_col=label_col)

    dataset = MemeDataset(df, is_train=is_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        drop_last=is_train,
        collate_fn=custom_collate_fn,
    )
    return loader


# ====================================================
# 🔄 Grouped k-Fold Splitter — Leak-Proof
# ====================================================
def make_grouped_folds(
    df,
    n_splits=5,
    group_col="template_id",
    stratify_col="label_sarcasm_intensity",
):
    """
    Create grouped stratified folds ensuring no template leakage and balanced label ratios.
    """
    if group_col not in df.columns or stratify_col not in df.columns:
        raise ValueError(f"Data must contain both '{group_col}' and '{stratify_col}' columns.")

    groups = df[group_col].values
    y = df[stratify_col].values

    print(
        f"\n🔧 Creating {n_splits}-fold GroupedStratified splits by '{group_col}' "
        f"(unique templates={df[group_col].nunique()})"
    )
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    folds = []
    for i, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups)):
        val_dist = df.iloc[val_idx][stratify_col].value_counts(normalize=True).to_dict()
        print(f"  📊 Fold {i}: {len(train_idx)} train / {len(val_idx)} val | dist={val_dist}")
        folds.append((train_idx, val_idx))

    print(
        f"✅ Created {n_splits}-fold grouped stratified splits "
        f"(unique templates: {df[group_col].nunique()}, samples: {len(df)})"
    )
    return folds
