# ====================================================
# train_main.py — Research-Grade SARC-MT-CLIP++ Training Pipeline (v2.4)
# ====================================================
"""
Conference-ready multimodal sarcasm training pipeline (v2.4):
  ✅ Automatic sarcasm column detection & normalization
  ✅ Balanced sampling + CORAL ordinal regression
  ✅ Cross-modal fusion & veracity-aware features
  ✅ SWA, cosine annealing, contrastive + incongruity losses
  ✅ Manual Model Summary Table
  ✅ Normalized, Labeled Confusion Matrix
  ✅ Loss/F1 Curves
"""
import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
)
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

import src.config as config
from src.dataset_module import create_data_loader, make_grouped_folds
from src.model_hmt import HierarchicalMultimodalTransformerCLIP

warnings.filterwarnings("ignore")

# ====================================================
# 🔁 Reproducibility
# ====================================================
def set_all_seeds(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(config.RANDOM_SEED)
scaler = GradScaler(enabled=config.USE_AMP)

# ====================================================
# 📉 Loss Functions
# ====================================================
def cb_focal_bce(logits, targets, gamma=config.FOCAL_GAMMA):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    pt = torch.where(targets > 0.5, probs, 1 - probs)
    return (bce * ((1 - pt) ** gamma)).mean()

def cb_focal_ce(logits, targets, gamma=config.FOCAL_GAMMA, class_weights=None):
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return (ce * ((1 - pt) ** gamma)).mean()

def coral_loss(logits, targets, class_weights=None):
    Kminus1 = logits.size(1)
    device = logits.device
    targets_expanded = targets.unsqueeze(1).repeat(1, Kminus1)
    thresholds = torch.arange(0, Kminus1, device=device).unsqueeze(0)
    target_mat = (targets_expanded > thresholds).float()
    loss = F.binary_cross_entropy_with_logits(logits, target_mat, reduction="none")
    if class_weights is not None:
        weights = class_weights[targets].unsqueeze(1).repeat(1, Kminus1)
        loss = loss * weights
    return loss.mean()

def coral_logits_to_preds(coral_logits):
    probs = torch.sigmoid(coral_logits)
    preds = (probs > 0.5).sum(dim=1)
    return preds.long()

def supervised_contrastive_loss(features, labels, temperature=config.CONTRASTIVE_TEMP):
    device = features.device
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask *= logits_mask
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()

def incongruity_margin_loss(sim, labels, margin=config.INCONGRUITY_MARGIN):
    is_sarc = (labels > 0).float()
    baseline = 1.0 - is_sarc
    return F.relu(sim.squeeze(1) - baseline + margin).mean()

def consistency_kl(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    return F.kl_div(p, q, reduction="batchmean")

# ====================================================
# ⚖️ Balanced Sampler
# ====================================================
def create_balanced_sampler(df, label_col="label_sarcasm_intensity"):
    class_counts = df[label_col].value_counts().sort_index().values
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = df[label_col].map({i: w for i, w in enumerate(weights)}).values
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ====================================================
# 📊 Evaluation
# ====================================================
def evaluate(model, data_loader, device, threshold_presence=0.5):
    model.eval()
    all_true, all_pred, all_pres_true, all_pres_pred = [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch["image_orig"].to(device)
            texts = batch["text_raw"]
            labels = batch["labels"]["label_sarcasm_intensity"].to(device)
            outs = model(text_raw=texts, pixel_values=imgs)
            int_logits = outs["sarcasm_intensity"]
            pres_logits = outs["sarcasm_presence"]
            preds_int = coral_logits_to_preds(int_logits)
            pres_probs = torch.sigmoid(pres_logits.squeeze())
            pres_pred = (pres_probs > threshold_presence).long()
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds_int.cpu().numpy())
            all_pres_true.append((labels > 0).long().cpu().numpy())
            all_pres_pred.append(pres_pred.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_pres_true = np.concatenate(all_pres_true)
    y_pres_pred = np.concatenate(all_pres_pred)
    return {
        "f1_macro_intensity": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy_intensity": accuracy_score(y_true, y_pred),
        "confusion_intensity": confusion_matrix(y_true, y_pred),
        "class_report_intensity": classification_report(y_true, y_pred, zero_division=0),
        "ordinal_mae": mean_absolute_error(y_true, y_pred),
        "presence_f1": f1_score(y_pres_true, y_pres_pred, average="binary", zero_division=0),
        "presence_acc": accuracy_score(y_pres_true, y_pres_pred),
    }

# ====================================================
# 🧠 Main Training Loop
# ====================================================
def train_main():
    print(f"\n🚀 Starting SARC-MT-CLIP++ training on {config.DEVICE}\n")
    set_all_seeds(config.RANDOM_SEED)

    df = pd.read_csv(config.LABEL_FILE)
    print(f"✅ Loaded {len(df)} samples from {config.LABEL_FILE}")
    print("📑 Columns detected:", df.columns.tolist())

    # --- Detect sarcasm intensity column ---
    possible_cols = ["label_sarcasm_intensity", "sarcasm_intensity", "sarcasm", "intensity", "sarcasm_label"]
    detected_col = next((col for col in possible_cols if col in df.columns), None)
    if not detected_col:
        raise ValueError(f"❌ No sarcasm intensity column found. Expected one of: {possible_cols}")
    if detected_col != "label_sarcasm_intensity":
        df.rename(columns={detected_col: "label_sarcasm_intensity"}, inplace=True)

    if df["label_sarcasm_intensity"].dtype == object:
        mapping = {"not_sarcastic": 0, "general": 1, "twisted_meaning": 2, "sarcastic": 1, "very_twisted": 2}
        df["label_sarcasm_intensity"] = df["label_sarcasm_intensity"].map(mapping).fillna(0).astype(int)

    if "template_id" not in df.columns:
        df["template_id"] = df["image_name"].apply(lambda x: os.path.splitext(str(x))[0])

    classes = np.unique(df["label_sarcasm_intensity"])
    class_weights_np = compute_class_weight("balanced", classes=classes, y=df["label_sarcasm_intensity"])
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(config.DEVICE)

    folds = make_grouped_folds(df, n_splits=config.N_FOLDS)
    train_idx, val_idx = folds[0]
    train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

    sampler = create_balanced_sampler(train_df)
    train_loader_base = create_data_loader(train_df, batch_size=config.BATCH_SIZE, is_train=True)
    val_loader_base = create_data_loader(val_df, batch_size=config.BATCH_SIZE, is_train=False)
    collate_fn = getattr(train_loader_base, "collate_fn", None)
    train_loader = DataLoader(train_loader_base.dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_loader_base.dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # --- Model setup ---
    model = HierarchicalMultimodalTransformerCLIP().to(config.DEVICE)

    # ====================================================
    # 📋 MANUAL MODEL SUMMARY (Safe for Multimodal Models)
    # ====================================================
    print("\n📋 MODEL SUMMARY (Parameter Breakdown):")
    print(f"{'Layer':<40} {'Params':<15} {'Trainable':<10}")
    print("-" * 70)
    total_params, trainable_params = 0, 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"{name:<40} {num_params:<15,} {str(param.requires_grad):<10}")
    print("-" * 70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {total_params - trainable_params:,}")
    print("-" * 70)

    optimizer = torch.optim.AdamW([
        {"params": model.fusion.parameters(), "lr": config.LR_FUSION},
        {"params": model.text_proj.parameters(), "lr": config.LR_FUSION},
        {"params": model.image_proj.parameters(), "lr": config.LR_FUSION},
        {"params": list(model.head_sarcasm_presence.parameters()) + list(model.head_sarcasm_intensity.parameters()), "lr": config.LR_HEADS},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    swa_model, swa_start = None, int(config.NUM_EPOCHS * 0.75)
    if config.USE_SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    else:
        swa_scheduler = None

    best_val, patience_counter = -1.0, 0
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_losses, val_f1s = [], []

    # ====================================================
    # 🔁 Training Loop
    # ====================================================
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            imgs = batch["image_orig"].to(config.DEVICE)
            texts = batch["text_raw"]
            labels = batch["labels"]["label_sarcasm_intensity"].to(config.DEVICE)
            presence_targets = (labels > 0).float()
            with autocast(enabled=config.USE_AMP):
                outs = model(text_raw=texts, pixel_values=imgs, return_features=True)
                pres_logits = outs["sarcasm_presence"]
                int_logits = outs["sarcasm_intensity"]
                features = outs["features"]
                loss_presence = cb_focal_bce(pres_logits.squeeze(), presence_targets)
                loss_intensity = coral_loss(int_logits, labels, class_weights=class_weights_tensor)
                loss_contrast = supervised_contrastive_loss(features, (labels > 0).long())
                try:
                    clip_sim = model.veracity_checker(texts, image_batch=imgs)[:, -1].unsqueeze(1)
                except Exception:
                    clip_sim = None
                loss_incon = incongruity_margin_loss(clip_sim, labels) if clip_sim is not None else 0
                total_loss = (
                    config.LOSS_WEIGHTS["sarcasm_presence"] * loss_presence +
                    config.LOSS_WEIGHTS["sarcasm_intensity"] * loss_intensity +
                    0.1 * (loss_contrast + loss_incon)
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(total_loss.item())
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})
        if scheduler:
            scheduler.step()
        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            if swa_scheduler:
                swa_scheduler.step()

        val_metrics = evaluate(model, val_loader, config.DEVICE)
        val_f1 = val_metrics['f1_macro_intensity']
        print(f"\n[Epoch {epoch}] TrainLoss={running_loss/len(train_loader):.4f} | ValF1={val_f1:.4f}")
        print(val_metrics.get("class_report_intensity", ""))
        train_losses.append(running_loss / len(train_loader))
        val_f1s.append(val_f1)

        if val_f1 > best_val:
            best_val, patience_counter = val_f1, 0
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best_model.pt"))
            print("💾 Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    print(f"\n✅ Training completed. Best macro-F1 (intensity): {best_val:.4f}")

    # ====================================================
    # 📈 Plot Loss & F1 Curves
    # ====================================================
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_f1s)+1), val_f1s, label="Val F1", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss & Validation F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "training_curves.png"))
    plt.close()

    # ====================================================
    # 📊 Confusion Matrix (Normalized + Labeled)
    # ====================================================
    cm = val_metrics["confusion_intensity"].astype(float)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    labels = ["Not Sarc", "Mild", "High"]
    plt.figure(figsize=(5,5))
    plt.imshow(cm_normalized, cmap="Blues")
    plt.title("Normalized Confusion Matrix (Sarcasm Intensity)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f"{cm_normalized[i, j]*100:.1f}%", ha="center", va="center", color="black")
    plt.colorbar(label="Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    train_main()
