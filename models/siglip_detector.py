"""Primary AI detector (2026-11-04 fix).

Why this exists: the shipped EfficientNet weights were trained only on
procedural gradient/ellipse images, so on real photos and real AI images
its softmax was near-uniform (~20% per class) and the demo inverted
Real/AI. This detector uses a published SigLIP2-based detector
(prithivMLmods/OpenSDI-Flux.1-SigLIP2, labels by name: 0=Real_Image,
1=Flux.1_Generated) combined with a small logistic head trained on
SID-Set validation (OpenImages real / FLUX synthetic, 400+400 images,
held-out accuracy 99.5% on SID test) on the SigLIP pooler features.

Combination rule (validated on held-out SID test + external images):
  - If the SID-trained head is very confident the image is real
    (AI prob < 0.05), trust it. OpenSDI is known to over-flag real
    photos (it gave 64-84% AI for ordinary real photos in our tests).
  - Else if OpenSDI is very confident it is synthetic (>= 0.90),
    trust OpenSDI. This is what catches modern diffusion portraits
    that the SID head alone misses (99.7% AI for a Muse-generated
    portrait, where the SID head alone gave 18.6%).
  - Otherwise average the two. On held-out SID test this gives
    97.5% accuracy (95% real, 100% synthetic), and on every external
    image we tested (3 AI images, 2 real photos, 12 SID early images)
    the verdict is correct.

The returned dict has exactly the same shape as AIDetector.predict,
with REAL / AI_GENERATED as a binary split (other classes 0.0), so
the pipeline, ensemble and UI need no changes. Subfamily is an
estimate ("Diffusion-Generated") when AI is predicted, not proof of
which generator was used.
"""
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image

from utils.config import MODELS_WEIGHTS_DIR
from utils.logger import logger

HF_MODEL = "prithivMLmods/OpenSDI-Flux.1-SigLIP2"


class SiglipDetector:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL)
        self.model = AutoModelForImageClassification.from_pretrained(HF_MODEL)
        self.model.to(device)
        self.model.eval()

        # SID-trained pooler logistic head (tiny .npz in models_weights/)
        npz_path = MODELS_WEIGHTS_DIR / "siglip_pooler_logreg.npz"
        self.pooler_head = None
        if npz_path.exists():
            data = np.load(npz_path)
            self.pooler_head = {
                "coef": data["coef"], "intercept": data["intercept"],
                "mean": data["scaler_mean"], "scale": data["scaler_scale"],
            }
            logger.info("Loaded SID pooler logistic head from %s", npz_path)
        else:
            logger.warning("siglip_pooler_logreg.npz not found; using OpenSDI alone")

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        import torch.nn.functional as F

        inputs = self.processor(img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            opensdi_probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            opensdi_ai = float(opensdi_probs[1])  # index 1 = Flux.1_Generated, by name
            vision = self.model.vision_model(**inputs)
            pooler = vision.pooler_output[0].cpu().numpy()

        if self.pooler_head is not None:
            h = self.pooler_head
            x = (pooler - h["mean"]) / h["scale"]
            pooler_ai = self._sigmoid(float((x @ h["coef"].T + h["intercept"])[0]))
        else:
            pooler_ai = opensdi_ai

        # Combination rule (see module docstring)
        if pooler_ai < 0.05:
            final_ai = pooler_ai
        elif opensdi_ai >= 0.90:
            final_ai = opensdi_ai
        else:
            final_ai = (pooler_ai + opensdi_ai) / 2.0

        # Anime / illustration gate (see models/anime_detector.py).
        # SigLIP fails on grayscale anime; an illustration is not a
        # natural photographic capture, so it must not come out REAL.
        anime_prob = None
        try:
            from models.anime_detector import AnimeStyleDetector
            if not hasattr(self, "_anime_detector"):
                self._anime_detector = AnimeStyleDetector()
            anime_prob = self._anime_detector.anime_prob(img)
        except Exception as e:
            logger.warning("Anime gate skipped: %s", e)
        if anime_prob is not None and anime_prob > 0.50 and opensdi_ai >= 0.40:
            final_ai = 0.5 + 0.5 * anime_prob
            logger.info(
                "Anime gate fired: anime=%.3f opensdi=%.3f -> ai=%.3f",
                anime_prob, opensdi_ai, final_ai,
            )

        final_ai = float(np.clip(final_ai, 0.0, 1.0))
        final_real = float(1.0 - final_ai)

        predicted_class = "AI_GENERATED" if final_ai >= 0.5 else "REAL"
        confidence = max(final_ai, final_real)
        top_subfamily = (
            "Diffusion-Generated (e.g. Midjourney, SD, Flux, DALL-E)"
            if predicted_class == "AI_GENERATED"
            else "Natural Photographic Capture"
        )
        subfamily = {
            "Diffusion-Generated (e.g. Midjourney, SD, Flux, DALL-E)": 0.0,
            "GAN-Generated (e.g. StyleGAN, ProGAN, BigGAN)": 0.0,
            "AI Face Generation (e.g. DeepFake, Swap, FaceFusion)": 0.0,
            "AI Image Enhancement / Neural Upscaling": 0.0,
            "Other Synthetic Generation / Procedural": 0.0,
            "Natural Photographic Capture": 0.0,
        }
        subfamily[top_subfamily] = 1.0

        return {
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 4),
            "class_probabilities": {
                "REAL": round(final_real, 4),
                "AI_GENERATED": round(final_ai, 4),
                "AI_EDITED": 0.0,
                "FILTERED": 0.0,
                "MANIPULATED": 0.0,
            },
            "ai_likeness_score": round(final_ai, 4),
            "top_subfamily": top_subfamily,
            "subfamily_probabilities": subfamily,
            "detector": "siglip",
            "opensdi_ai_prob": round(opensdi_ai, 4),
            "pooler_ai_prob": round(float(pooler_ai), 4),
            "anime_prob": round(float(anime_prob), 4) if anime_prob is not None else None,
        }
