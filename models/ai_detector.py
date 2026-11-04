from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

from utils.config import (
    PRIMARY_CLASSES, PRIMARY_CLASS_TO_IDX, IDX_TO_PRIMARY_CLASS,
    AI_SUBFAMILIES, DEVICE, AI_MODEL_PATH
)
from utils.logger import logger
from preprocessing.image_processor import ImageProcessor


class AIDetectionNetwork(nn.Module):
    """
    Deep Vision Network for AI-Generated Image Detection.
    Built on transfer-learning backbone (EfficientNet-B0) with dual classification heads:
    1. Authenticity Classifier (Real, AI-Generated, AI-Edited, Filtered, Manipulated)
    2. AI Subfamily Estimator (Diffusion, GAN, DeepFake/Face, Enhancement, Other)
    """

    def __init__(self, backbone_name: str = "efficientnet_b0", num_classes: int = len(PRIMARY_CLASSES), num_subfamilies: int = len(AI_SUBFAMILIES)):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_subfamilies = num_subfamilies

        # Load backbone
        if backbone_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.features = base_model.features
            in_features = base_model.classifier[1].in_features
            self.target_layer = self.features[-1] # For Grad-CAM hook
        elif backbone_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            in_features = base_model.fc.in_features
            self.target_layer = list(self.features.children())[-1]
        else:
            # Fallback to efficientnet_b0
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.features = base_model.features
            in_features = base_model.classifier[1].in_features
            self.target_layer = self.features[-1]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Primary Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        # AI Subfamily Head
        self.subfamily_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_subfamilies)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        pooled = self.pool(feat)
        flattened = torch.flatten(pooled, 1)

        primary_logits = self.classifier(flattened)
        subfamily_logits = self.subfamily_head(flattened)

        return primary_logits, subfamily_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns convolutional feature map for explainability."""
        return self.features(x)


class AIDetector:
    """Inference wrapper for the AI Generation Detection Network."""

    def __init__(self, model_path: Optional[str] = None, device: str = DEVICE):
        self.device = device
        self.processor = ImageProcessor()
        self.model = AIDetectionNetwork().to(self.device)
        self.model.eval()

        path = model_path if model_path else AI_MODEL_PATH
        if path and path.exists():
            try:
                try:
                    state = torch.load(path, map_location=self.device, weights_only=False)
                except TypeError:
                    state = torch.load(path, map_location=self.device)
                if "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                logger.info(f"Loaded trained AI Detector weights from {path}")
            except Exception as e:
                logger.warning(f"Could not load custom weights from {path}: {e}. Running with pretrained vision backbone.")
        else:
            logger.info("Running AI Detector with pretrained vision backbone.")

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        """
        Runs deep learning inference on an input image.
        Returns class probabilities, predicted category, subfamily estimation, and confidence.
        Primary detector (2026-11-04): SigLIP (see models/siglip_detector.py).
        EfficientNet below is the fallback if SigLIP cannot load, and its
        model is kept for Grad-CAM.
        """
        # Lazy SigLIP load on first prediction (so app startup is not
        # blocked by the ~350MB model download; analysis runs in a spinner).
        if not hasattr(self, "_siglip_tried"):
            self._siglip_tried = True
            self._siglip = None
            try:
                from models.siglip_detector import SiglipDetector
                self._siglip = SiglipDetector(device=self.device)
                logger.info("Using SigLIP primary detector")
            except Exception as e:
                logger.warning(f"SigLIP detector unavailable, falling back to EfficientNet: {e}")
        if getattr(self, "_siglip", None) is not None:
            return self._siglip.predict(img)

        tensor = self.processor.to_tensor(img, device=self.device)

        with torch.no_grad():
            primary_logits, subfamily_logits = self.model(tensor)
            primary_probs = F.softmax(primary_logits, dim=1).squeeze(0).cpu().numpy()
            subfamily_probs = F.softmax(subfamily_logits, dim=1).squeeze(0).cpu().numpy()

        class_probs = {
            cls: float(primary_probs[idx])
            for idx, cls in enumerate(PRIMARY_CLASSES)
        }

        # Identify top predicted class directly from the model softmax.
        # NOTE (2026-11-04 fix): a previous "natural-photo guardrail" forced
        # predicted_class to REAL whenever REAL was merely >= AI_GENERATED
        # (even at ~0.25) and capped ai_likeness at 0.20. That made the
        # predicted class, the class probabilities and the AI-likeness
        # disagree with each other, and made uncertain AI images report as
        # Real. The verdict must follow the probabilities, not override them.
        top_idx = int(np.argmax(primary_probs))
        predicted_class = PRIMARY_CLASSES[top_idx]
        confidence = float(primary_probs[top_idx])
        top_subfamily_idx = int(np.argmax(subfamily_probs))
        top_subfamily = AI_SUBFAMILIES[top_subfamily_idx]

        # Subfamily predictions
        subfamily_dict = {
            AI_SUBFAMILIES[idx]: float(subfamily_probs[idx])
            for idx in range(len(AI_SUBFAMILIES))
        }

        if predicted_class == "REAL":
            top_subfamily = "Natural Photographic Capture"
            subfamily_dict = {key: 0.0 for key in subfamily_dict}
            subfamily_dict[top_subfamily] = 1.0

        # Calculate binary AI-likeness (sum of AI_GENERATED, AI_EDITED, MANIPULATED vs REAL).
        # Do NOT cap it when the prediction is REAL: capping it hid the
        # model's own synthetic signal and broke the ensemble downstream.
        ai_likeness = float(class_probs.get("AI_GENERATED", 0.0) + 0.6 * class_probs.get("AI_EDITED", 0.0) + 0.4 * class_probs.get("MANIPULATED", 0.0))
        ai_likeness = min(1.0, max(0.0, ai_likeness))

        return {
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 4),
            "class_probabilities": {k: round(v, 4) for k, v in class_probs.items()},
            "ai_likeness_score": round(ai_likeness, 4),
            "top_subfamily": top_subfamily,
            "subfamily_probabilities": {k: round(v, 4) for k, v in subfamily_dict.items()}
        }
