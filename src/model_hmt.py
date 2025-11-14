# ====================================================
# src/model_hmt.py — Fully Updated SARC-MT-CLIP++ Model (Research-Grade)
# ====================================================
"""
Hierarchical Multimodal Transformer (SARC-MT-CLIP++)

- Frozen CLIP encoders (with late unfreezing)
- Dual projection heads (image/text)
- Tiny cross-modal transformer fusion (2 layers)
- Veracity (NLI) module integration
- Multi-task heads (sarcasm + auxiliary)
- Optional uncertainty weighting (log_vars)
- Supports feature extraction & joint embeddings
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from open_clip import create_model_and_transforms, get_tokenizer
from .veracity_checker import VeracityChecker
from .config import (
    FUSION_HIDDEN_DIM,
    FUSION_DROPOUT,
    CLASSIFIER_DROPOUT,
    VERACITY_OUTPUT_DIM,
    DEVICE,
    RANDOM_SEED,
    FREEZE_CLIP,
    UNFREEZE_LAST_BLOCKS,
    USE_DYNAMIC_UNCERTAINTY,
)

torch.manual_seed(RANDOM_SEED)


# ====================================================
# 🔁 Cross-Modal Transformer Fusion
# ====================================================
class CrossModalFusion(nn.Module):
    """
    Lightweight transformer-based fusion between CLIP text and image embeddings.
    """

    def __init__(
        self,
        dim: int = FUSION_HIDDEN_DIM,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = FUSION_DROPOUT,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        # Combine both modalities as sequence [B, 2, D]
        x = torch.stack([img_emb, txt_emb], dim=1)
        fused = self.encoder(x)
        pooled = fused.mean(dim=1)  # mean-pool across modalities
        return self.norm(pooled)


# ====================================================
# 🚀 Hierarchical Multimodal Transformer (SARC-MT-CLIP++)
# ====================================================
class HierarchicalMultimodalTransformerCLIP(nn.Module):
    def __init__(
        self,
        clip_backbone: str = "ViT-B-32",
        pretrained_clip: str = "openai",
        num_intensity_classes: int = 3,
        hidden_dim: int = FUSION_HIDDEN_DIM,
        classifier_dropout: float = CLASSIFIER_DROPOUT,
        freeze_clip: bool = FREEZE_CLIP,
        unfreeze_last_blocks: int = UNFREEZE_LAST_BLOCKS,
        veracity_output_dim: int = VERACITY_OUTPUT_DIM,
    ):
        super().__init__()
        self.device = DEVICE
        self.hidden_dim = hidden_dim
        self.num_intensity_classes = num_intensity_classes
        self.intensity_kminus1 = num_intensity_classes - 1

        # ---------------------
        # 🧠 CLIP backbone
        # ---------------------
        self.clip_model, _, _ = create_model_and_transforms(
            clip_backbone, pretrained=pretrained_clip
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = get_tokenizer(clip_backbone)

        # Freeze or selectively unfreeze CLIP
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False
        self._clip_frozen = freeze_clip
        self.unfreeze_last_blocks = unfreeze_last_blocks

        # Infer CLIP embedding dimension
        clip_emb_dim = getattr(self.clip_model, "text_projection", None)
        clip_emb_dim = (
            int(self.clip_model.text_projection.shape[1])
            if clip_emb_dim is not None
            else 512
        )

        # ---------------------
        # 🔧 Projection heads
        # ---------------------
        self.text_proj = nn.Sequential(
            nn.Linear(clip_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(clip_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---------------------
        # 🧩 Veracity Checker (NLI model)
        # ---------------------
        self.veracity_checker = VeracityChecker(output_dim=veracity_output_dim)

        # ---------------------
        # 🔀 Fusion & Regularization
        # ---------------------
        self.fusion = CrossModalFusion(dim=hidden_dim)
        self.dropout = nn.Dropout(classifier_dropout)

        # ---------------------
        # 🧠 Joint feature layer norm
        # ---------------------
        joint_dim = hidden_dim + veracity_output_dim + 1
        self.norm = nn.LayerNorm(joint_dim)

        def make_head(out_dim: int):
            return nn.Sequential(
                nn.Linear(joint_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        # ---------------------
        # 🎯 Multi-task Output Heads
        # ---------------------
        self.head_sarcasm_presence = make_head(1)
        self.head_sarcasm_intensity = make_head(self.intensity_kminus1)
        self.head_humor = make_head(3)
        self.head_offense = make_head(3)
        self.head_sentiment = make_head(3)
        self.head_motivation = make_head(4)

        # ---------------------
        # ⚖️ Optional Uncertainty Weights
        # ---------------------
        if USE_DYNAMIC_UNCERTAINTY:
            self.log_vars = nn.Parameter(torch.zeros(3, device=self.device), requires_grad=True)
        else:
            self.log_vars = None

        self.to(self.device)

    # ====================================================
    # 🧾 Tokenization Helper
    # ====================================================
    def _tokenize_texts(self, text_list: List[str]):
        tokens = self.clip_tokenizer(text_list)
        if isinstance(tokens, dict):
            return tokens["input_ids"].to(self.device)
        elif isinstance(tokens, torch.Tensor):
            return tokens.to(self.device)
        else:
            return torch.tensor(tokens).to(self.device)

    # ====================================================
    # 🔍 CLIP Encoding
    # ====================================================
    @torch.no_grad()
    def _encode_clip(
        self, text_list: List[str], images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        txt_tokens = self._tokenize_texts(text_list)
        txt_emb = self.clip_model.encode_text(txt_tokens)
        img_emb = self.clip_model.encode_image(images)
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        return img_emb, txt_emb

    # ====================================================
    # 🔄 Unfreeze last CLIP blocks (optional fine-tuning)
    # ====================================================
    def unfreeze_last_clip_blocks(self):
        """
        Gradually unfreeze last CLIP transformer blocks for fine-tuning.
        """
        if hasattr(self.clip_model, "visual") and hasattr(self.clip_model.visual, "transformer"):
            blocks = list(self.clip_model.visual.transformer.resblocks)
            for block in blocks[-self.unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        if hasattr(self.clip_model, "transformer"):
            blocks = list(self.clip_model.transformer.resblocks)
            for block in blocks[-self.unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        self._clip_frozen = False
        print(f"✅ Unfroze last {self.unfreeze_last_blocks} CLIP transformer blocks.")

    # ====================================================
    # 🔥 Forward
    # ====================================================
    def forward(
        self,
        text_raw: List[str],
        pixel_values: torch.Tensor,
        return_features: bool = False,
        return_veracity: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # ---------------------
        # CLIP embeddings
        # ---------------------
        if self._clip_frozen:
            with torch.no_grad():
                img_emb, txt_emb = self._encode_clip(text_raw, pixel_values)
        else:
            img_emb, txt_emb = self._encode_clip(text_raw, pixel_values)

        # ---------------------
        # Projection and fusion
        # ---------------------
        img_feat = self.image_proj(img_emb)
        txt_feat = self.text_proj(txt_emb)
        fused = self.fusion(img_feat, txt_feat)

        batch_size = pixel_values.size(0)

        # ---------------------
        # Veracity + similarity
        # ---------------------
        if return_veracity:
            try:
                ver_feats = self.veracity_checker(text_raw, image_batch=pixel_values)
                if ver_feats.size(1) > VERACITY_OUTPUT_DIM:
                    ver_feats = ver_feats[:, :VERACITY_OUTPUT_DIM]
                clip_sim = torch.sum(img_emb * txt_emb, dim=-1, keepdim=True)
            except Exception:
                ver_feats = torch.zeros((batch_size, VERACITY_OUTPUT_DIM), device=self.device)
                clip_sim = torch.sum(img_emb * txt_emb, dim=-1, keepdim=True)
        else:
            ver_feats = torch.zeros((batch_size, VERACITY_OUTPUT_DIM), device=self.device)
            clip_sim = torch.zeros((batch_size, 1), device=self.device)

        # ---------------------
        # Joint feature fusion
        # ---------------------
        joint = torch.cat([fused, ver_feats, clip_sim], dim=1)
        joint = self.norm(self.dropout(joint))

        outputs = {
            "sarcasm_presence": self.head_sarcasm_presence(joint),
            "sarcasm_intensity": self.head_sarcasm_intensity(joint),
            "humor": self.head_humor(joint),
            "offense": self.head_offense(joint),
            "sentiment": self.head_sentiment(joint),
            "motivation": self.head_motivation(joint),
            "sim": clip_sim,
            "image_embeds": img_emb,
            "text_embeds": txt_emb,
        }

        if return_features:
            outputs["features"] = joint

        return outputs
