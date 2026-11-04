"""Anime / illustration style detector.

Why: SigLIP (OpenSDI + SID pooler head) was trained only on
photorealistic photos vs photorealistic Flux images. It fails on
anime/manga-style AI images: a grayscale anime portrait from the user
got OpenSDI 50.6% AI but pooler 0.05% AI, so the combined rule called
it REAL 99.95%. Colour anime works via OpenSDI alone, grayscale anime
does not.

This detector is deepghs/anime_real_cls (MobileNetV3, ONNX, 17 MB),
labels [anime, real]. Its raw ONNX output is already a probability
vector (NOT logits — do not softmax it again; softmaxing the
probabilities collapses everything to ~72%/27%).

Gate rule (validated 2026-11-04): only override SigLIP when BOTH
  anime_prob > 0.50  AND  opensdi_ai_prob >= 0.40
The second condition matters: on held-out SID real photos, 8/100 have
anime_prob > 0.50 (ordinary photos the anime model misreads), but
their OpenSDI AI prob is 0.03-0.19, so the gate does not fire for
them. For the 5 anime test images (user grayscale portrait, colour
anime, grayscale boy/cat/city) anime_prob is 0.55-0.98 and OpenSDI
is 0.41-1.00, so the gate fires for all 5.

When the gate fires, the image is an illustration, not a natural
photographic capture, so REAL (photographic) cannot be the verdict.
AI prob is mapped as 0.5 + 0.5*anime_prob (user image: 0.99,
city: 0.77) — high but still monotonic in the anime detector's
confidence, not a flat 1.0.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from utils.config import MODELS_WEIGHTS_DIR
from utils.logger import logger


class AnimeStyleDetector:
    def __init__(self):
        self.session = None
        onnx_path = MODELS_WEIGHTS_DIR / "anime_real_cls.onnx"
        if not onnx_path.exists():
            logger.warning("anime_real_cls.onnx not found; anime gate disabled")
            return
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info("Loaded anime style detector from %s", onnx_path)
        except Exception as e:
            logger.warning("Anime style detector failed to load: %s", e)
            self.session = None

    def anime_prob(self, img: Image.Image) -> Optional[float]:
        if self.session is None:
            return None
        try:
            resized = img.convert("RGB").resize((384, 384))
            arr = np.array(resized).astype(np.float32) / 255.0
            x = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
            out = self.session.run(None, {self.input_name: x})[0]
            # Output is already [anime_prob, real_prob]; do NOT softmax.
            return float(out[0][0])
        except Exception as e:
            logger.warning("Anime style detector inference failed: %s", e)
            return None
