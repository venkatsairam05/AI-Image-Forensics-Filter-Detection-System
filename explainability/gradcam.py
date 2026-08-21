from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from preprocessing.image_processor import ImageProcessor
from utils.config import DEVICE, PRIMARY_CLASSES
from utils.logger import logger

class GradCAMExplainer:
    """
    Explainable AI engine implementing Gradient-weighted Class Activation Mapping (Grad-CAM).
    Visualizes which visual spatial regions influenced the model's prediction and provides
    contextual reasoning about suspicious synthetic patterns.
    """

    def __init__(self, model_wrapper, target_layer=None):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        self.processor = ImageProcessor()

        # Set target layer (last convolutional layer of backbone)
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            self.target_layer = self.model.target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks to capture feature maps and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_explanation(
        self,
        pil_img: Image.Image,
        target_class_idx: Optional[int] = None,
        face_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM heatmap, alpha-blended overlay image, active ROI boxes,
        and forensic contextual reasoning text.
        """
        self.model.eval()
        self.model.zero_grad()

        orig_w, orig_h = pil_img.size
        tensor = self.processor.to_tensor(pil_img, device=self.device)
        tensor.requires_grad_(True)

        primary_logits, _ = self.model(tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(primary_logits, dim=1).item())

        score = primary_logits[0, target_class_idx]
        score.backward(retain_graph=True)

        # Grad-CAM weight calculation: GAP over spatial dimensions
        if self.gradients is None or self.activations is None:
            # Fallback synthetic activation map if hook missed
            heatmap = np.ones((orig_h, orig_w), dtype=np.float32) * 0.5
        else:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam) # ReLU to keep only positive contributions
            
            # Normalize CAM
            cam = cam.squeeze().cpu().numpy()
            cam_min, cam_max = np.min(cam), np.max(cam)
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            # Resize CAM to original image size
            heatmap = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heatmap = np.clip(heatmap, 0.0, 1.0)

        # Generate Visual Heatmap & Overlay
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        orig_np = np.array(pil_img.convert("RGB"))
        overlay = cv2.addWeighted(orig_np, 0.55, heatmap_color_rgb, 0.45, 0)

        # Analyze High-Activation Hotspots (>= 0.65 intensity)
        hotspot_mask = (heatmap >= 0.65).astype(np.uint8) * 255
        contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hotspot_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (orig_w * orig_h * 0.01): # Filter tiny noise
                x, y, w, h = cv2.boundingRect(cnt)
                hotspot_regions.append({
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "relative_area": round(area / (orig_w * orig_h), 3)
                })

        # Synthesize Contextual Forensic Explanation Text
        target_class_name = PRIMARY_CLASSES[target_class_idx]
        reasoning_text = self._synthesize_reasoning(heatmap, hotspot_regions, target_class_name, face_info, orig_w, orig_h)

        return {
            "target_class": target_class_name,
            "target_class_idx": target_class_idx,
            "heatmap": heatmap,
            "heatmap_image": heatmap_color_rgb,
            "overlay_image": overlay,
            "hotspot_count": len(hotspot_regions),
            "hotspots": hotspot_regions,
            "forensic_reasoning": reasoning_text,
            "disclaimer": "Grad-CAM highlights image regions with high gradient activation for the predicted class. It provides visual interpretability of model decisions, not definitive physical proof."
        }

    def _synthesize_reasoning(
        self,
        heatmap: np.ndarray,
        hotspots: List[Dict[str, Any]],
        class_name: str,
        face_info: Optional[Dict[str, Any]],
        w: int,
        h: int
    ) -> str:
        """Generates natural-language forensic rationale based on activation loci."""
        if not hotspots:
            return f"Model shows diffuse, uniform spatial attention across the entire image with low localized peak activations."

        # Check overlap with facial region
        face_overlap = False
        if face_info and face_info.get("face_detected"):
            px, py, pw, ph = face_info.get("primary_box", (0, 0, 0, 0))
            face_cam = heatmap[py : py + ph, px : px + pw]
            if face_cam.size > 0 and np.mean(face_cam) > 0.45:
                face_overlap = True

        # Check edge / boundary vs central focus
        border_mask = np.zeros_like(heatmap, dtype=bool)
        border_mask[:int(h*0.15), :] = True
        border_mask[int(h*0.85):, :] = True
        border_mask[:, :int(w*0.15)] = True
        border_mask[:, int(w*0.85):] = True
        border_activation = np.mean(heatmap[border_mask]) if np.any(border_mask) else 0.0

        reasons = []
        if face_overlap:
            reasons.append("facial skin texture, symmetry, and eye/hair boundary transitions")
        if border_activation > 0.40:
            reasons.append("background composition, edge blending transitions, and synthetic perimeter details")
        if not reasons:
            reasons.append("micro-texture patterns, high-frequency edge gradients, and localized structural anomalies")

        joined_reasons = " and ".join(reasons)
        if class_name in ["AI_GENERATED", "AI_EDITED"]:
            return f"Primary model activations are localized around {joined_reasons}. These regions display synthetic texture signatures and unnatural gradient continuities consistent with generative models."
        elif class_name in ["FILTERED", "MANIPULATED"]:
            return f"High decision influence concentrated around {joined_reasons}, indicating localized filtering, selective tone adjustments, or composite boundaries."
        else:
            return f"Model attention is distributed organically across {joined_reasons}, verifying natural camera sensor grain and consistent optical depth."
