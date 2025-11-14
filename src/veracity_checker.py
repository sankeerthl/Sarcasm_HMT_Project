# ====================================================
# src/veracity_checker.py — Research-Grade CLIP-aware Veracity & Incongruity Module (SARC-MT-CLIP++)
# ====================================================
"""
Produces multimodal veracity / incongruity representations that quantify
semantic disagreement between meme text and image.

Includes:
  - CLIP image-text similarity (frozen)
  - NLI (Natural Language Inference) signal via RoBERTa-large-MNLI
  - SBERT semantic fallback when NLI uncertain
  - Projection to low-dimensional veracity embedding
  - Output used in sarcasm & incongruity margin loss
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from open_clip import create_model_and_transforms, get_tokenizer
from .config import DEVICE, NLI_MODEL_NAME, VERACITY_OUTPUT_DIM


# ====================================================
# 🚀 CLIP + NLI + SBERT Veracity Checker
# ====================================================
class VeracityChecker(nn.Module):
    """
    Combines multimodal (CLIP) and textual (NLI/SBERT) veracity signals
    into a compact embedding used to estimate incongruity.
    """

    def __init__(self, output_dim: int = VERACITY_OUTPUT_DIM):
        super().__init__()
        self.device = DEVICE
        self.output_dim = output_dim

        # ----------------------------
        # CLIP backbone
        # ----------------------------
        try:
            self.clip_model, _, _ = create_model_and_transforms("ViT-B-32", pretrained="openai")
            self.clip_model = self.clip_model.to(self.device).eval()
            self.clip_tokenizer = get_tokenizer("ViT-B-32")
            self.clip_loaded = True
            print("✅ CLIP backbone loaded for veracity checking.")
        except Exception as e:
            print(f"⚠️ CLIP load failed: {e}")
            self.clip_model = None
            self.clip_tokenizer = None
            self.clip_loaded = False

        # ----------------------------
        # NLI-based veracity model
        # ----------------------------
        try:
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(self.device)
            self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            self.nli_loaded = True
            print(f"✅ NLI model '{NLI_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"⚠️ NLI model load failed: {e}. Falling back to SBERT only.")
            self.nli_loaded = False
            self.nli_model = None
            self.nli_tokenizer = None

        # ----------------------------
        # SBERT fallback model
        # ----------------------------
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        sbert_dim = self.sbert_model.get_sentence_embedding_dimension()
        self.sbert_proj = nn.Sequential(
            nn.Linear(sbert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # mimic NLI logits
        )

        # ----------------------------
        # Veracity projection head
        # ----------------------------
        self.veracity_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

        # ----------------------------
        # Hardcoded claim mappings for common meme contexts
        # ----------------------------
        self.hardcoded_claims = {
            "this is fine": "The situation is dangerous and chaotic.",
            "i love it": "The person actually hates it.",
            "great job": "The result is terrible.",
            "nothing to worry": "The situation is clearly concerning.",
            "best team": "The team performed very badly.",
            "perfect": "Something has gone wrong.",
        }

    # ====================================================
    # 🔍 Compute Veracity Embedding
    # ====================================================
    @torch.no_grad()
    def forward(self, text_batch: list[str], image_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Inputs:
          text_batch: list of raw meme texts
          image_batch: [B, 3, H, W] or None
        Returns:
          [B, VERACITY_OUTPUT_DIM + 1] tensor
          (+1 = CLIP cosine similarity scalar)
        """
        batch_size = len(text_batch)
        text_batch = [str(t).strip() for t in text_batch]

        # ----------------------------
        # 1️⃣ SBERT baseline signal
        # ----------------------------
        sbert_emb = self.sbert_model.encode(
            text_batch, convert_to_tensor=True, show_progress_bar=False
        )
        pseudo_logits = self.sbert_proj(sbert_emb)
        pseudo_probs = torch.softmax(pseudo_logits, dim=1)

        # ----------------------------
        # 2️⃣ NLI veracity scores
        # ----------------------------
        nli_probs = []
        for idx, text in enumerate(text_batch):
            claim = None
            for k in self.hardcoded_claims:
                if k in text.lower():
                    claim = self.hardcoded_claims[k]
                    break

            if self.nli_loaded and claim:
                try:
                    inputs = self.nli_tokenizer(
                        text,
                        claim,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    ).to(self.device)
                    logits = self.nli_model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    nli_probs.append(probs)
                    continue
                except Exception:
                    pass
            nli_probs.append(pseudo_probs[idx])

        nli_tensor = torch.stack(nli_probs).to(self.device)  # [B, 3]

        # ----------------------------
        # 3️⃣ Project to veracity embedding
        # ----------------------------
        ver_feats = self.veracity_proj(nli_tensor)  # [B, VERACITY_OUTPUT_DIM]

        # ----------------------------
        # 4️⃣ CLIP image–text similarity (cosine)
        # ----------------------------
        if self.clip_loaded and image_batch is not None:
            try:
                tokens = self.clip_tokenizer(text_batch).to(self.device)
                txt_feat = self.clip_model.encode_text(tokens)
                img_feat = self.clip_model.encode_image(image_batch)
                txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
                img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
                clip_sim = torch.sum(txt_feat * img_feat, dim=-1, keepdim=True)
            except Exception:
                clip_sim = torch.zeros(batch_size, 1, device=self.device)
        else:
            clip_sim = torch.zeros(batch_size, 1, device=self.device)

        # ----------------------------
        # 5️⃣ Combine all veracity signals
        # ----------------------------
        out = torch.cat([ver_feats, clip_sim], dim=1)
        return out


# ====================================================
# 🧪 Sanity Test
# ====================================================
if __name__ == "__main__":
    checker = VeracityChecker()
    dummy_imgs = torch.randn(3, 3, 224, 224).to(DEVICE)
    texts = ["This is fine", "Great job", "Nothing to worry about"]
    out = checker(texts, dummy_imgs)
    print("✅ Output shape:", out.shape)
    print("Example vector:", out[0][:10])
