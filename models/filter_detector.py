from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from utils.config import (
    FILTER_CLASSES, FILTER_THRESHOLDS, FILTER_CLASS_DESCRIPTIONS,
    DEVICE, FILTER_MODEL_PATH
)
from utils.logger import logger
from preprocessing.image_processor import ImageProcessor

class FilterDetectionNetwork(nn.Module):
    """
    Multi-Label Deep Vision Network for simultaneous image filter and transformation detection.
    """

    def __init__(self, num_filters: int = len(FILTER_CLASSES)):
        super().__init__()
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = base_model.features
        in_features = base_model.classifier[1].in_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits


class FilterDetector:
    """Inference wrapper for Multi-Label Filter Detection."""

    def __init__(self, model_path: Optional[str] = None, device: str = DEVICE):
        self.device = device
        self.processor = ImageProcessor()
        self.model = FilterDetectionNetwork().to(self.device)
        self.model.eval()

        path = model_path if model_path else FILTER_MODEL_PATH
        if path and path.exists():
            try:
                state = torch.load(path, map_location=self.device)
                if "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                logger.info(f"Loaded trained Filter Detector weights from {path}")
            except Exception as e:
                logger.warning(f"Could not load custom filter weights from {path}: {e}. Using baseline.")
        else:
            logger.info("Running Filter Detector with baseline vision weights.")

    def predict(self, img: Image.Image, forensic_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs multi-label filter prediction and combines CNN logits with forensic heuristics.
        """
        tensor = self.processor.to_tensor(img, device=self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        filter_scores = {}
        detected_filters = []

        for idx, filter_name in enumerate(FILTER_CLASSES):
            raw_score = float(probs[idx])
            
            # Incorporate CV forensic signals if provided
            if forensic_signals:
                if filter_name == "sharpening" and "texture_analysis" in forensic_signals:
                    lap_sharp = forensic_signals["texture_analysis"].get("laplacian_sharpness", 0.0)
                    if lap_sharp > 800:
                        raw_score = max(raw_score, min(0.95, lap_sharp / 1500.0))
                
                elif filter_name == "skin_smoothing" and "face_analysis" in forensic_signals:
                    skin_smooth = forensic_signals["face_analysis"].get("skin_smoothing_score", 0.0)
                    if skin_smooth > 0.4:
                        raw_score = max(raw_score, skin_smooth)

                elif filter_name == "compression_artifacts" and "compression_analysis" in forensic_signals:
                    grid = forensic_signals["compression_analysis"].get("jpeg_grid_strength", 0.0)
                    if grid > 0.5:
                        raw_score = max(raw_score, min(0.95, grid / 1.5))

            score_rounded = round(float(np.clip(raw_score, 0.0, 1.0)), 4)
            filter_scores[filter_name] = score_rounded

            thresh = FILTER_THRESHOLDS.get(filter_name, 0.50)
            if score_rounded >= thresh:
                detected_filters.append({
                    "name": filter_name,
                    "label": filter_name.replace("_", " ").title(),
                    "score": score_rounded,
                    "threshold": thresh,
                    "description": FILTER_CLASS_DESCRIPTIONS.get(filter_name, "")
                })

        # Sort detected filters by score descending
        detected_filters.sort(key=lambda x: x["score"], reverse=True)

        return {
            "all_filter_scores": filter_scores,
            "detected_filters": detected_filters,
            "detected_filter_names": [f["name"] for f in detected_filters],
            "filter_count": len(detected_filters),
            "max_filter_score": max(filter_scores.values()) if filter_scores else 0.0
        }
