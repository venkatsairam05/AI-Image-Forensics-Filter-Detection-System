from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image

class TextureAnalyzer:
    """
    Analyzes micro-texture distributions, Local Binary Patterns (LBP),
    and spatial Gray-Level Co-occurrence metrics to detect synthetic skin smoothing,
    excessive unsharp filtering, or abnormal texture homogeny.
    """

    def analyze(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Performs LBP, GLCM approximation, and gradient edge variance analysis."""
        gray = np.array(pil_img.convert("L"), dtype=np.uint8)
        
        # 1. Edge and Gradient Distribution via Sobel & Laplacian
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        mean_edge_gradient = float(np.mean(grad_mag))
        edge_gradient_std = float(np.std(grad_mag))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # 2. Local Binary Patterns (LBP) 8-neighborhood
        lbp_img = self._compute_lbp(gray)
        lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=32, range=(0, 256), density=True)
        lbp_entropy = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-12)))

        # 3. Spatial Co-occurrence Matrix (GLCM) approximation (horizontal offset d=1)
        glcm_metrics = self._compute_fast_glcm(gray)

        # 4. Compute Texture Anomaly Score
        texture_anomaly = self._compute_texture_anomaly(
            glcm_metrics["homogeneity"],
            glcm_metrics["contrast"],
            lbp_entropy,
            laplacian_var
        )

        return {
            "mean_edge_gradient": round(mean_edge_gradient, 4),
            "edge_gradient_std": round(edge_gradient_std, 4),
            "laplacian_sharpness": round(laplacian_var, 4),
            "lbp_entropy": round(lbp_entropy, 4),
            "glcm_contrast": round(glcm_metrics["contrast"], 4),
            "glcm_homogeneity": round(glcm_metrics["homogeneity"], 4),
            "glcm_energy": round(glcm_metrics["energy"], 4),
            "texture_anomaly_score": round(texture_anomaly, 4)
        }

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Fast vectorized Local Binary Pattern calculation (8-neighbor)."""
        h, w = gray.shape
        if h < 3 or w < 3:
            return gray
        
        center = gray[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)

        # 8 neighbors
        neighbors = [
            gray[:-2, :-2],  # top-left
            gray[:-2, 1:-1], # top
            gray[:-2, 2:],   # top-right
            gray[1:-1, 2:],  # right
            gray[2:, 2:],    # bottom-right
            gray[2:, 1:-1],  # bottom
            gray[2:, :-2],   # bottom-left
            gray[1:-1, :-2]  # left
        ]

        for i, neighbor in enumerate(neighbors):
            lbp += ((neighbor >= center).astype(np.uint8) << i)

        return lbp

    def _compute_fast_glcm(self, gray: np.ndarray, levels: int = 32) -> Dict[str, float]:
        """Computes quantized Gray-Level Co-occurrence Matrix statistics."""
        # Quantize to 32 levels for fast matrix calculation
        quantized = (gray // (256 // levels)).astype(np.int32)
        h, w = quantized.shape
        
        # Horizontal adjacent pair matrix (d=1, theta=0)
        p1 = quantized[:, :-1].ravel()
        p2 = quantized[:, 1:].ravel()

        glcm = np.zeros((levels, levels), dtype=np.float32)
        np.add.at(glcm, (p1, p2), 1)
        
        # Normalize GLCM
        total = np.sum(glcm)
        if total > 0:
            glcm /= total

        # Compute GLCM statistical descriptors
        i_idx, j_idx = np.indices((levels, levels))
        
        contrast = float(np.sum(glcm * ((i_idx - j_idx) ** 2)))
        homogeneity = float(np.sum(glcm / (1.0 + np.abs(i_idx - j_idx))))
        energy = float(np.sum(glcm ** 2))

        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy
        }

    def _compute_texture_anomaly(self, homogeneity: float, contrast: float, lbp_entropy: float, lap_var: float) -> float:
        """Estimates texture anomaly score from synthetic smoothness or over-sharpening."""
        score = 0.0
        # Over-smoothed skin / AI texture has very high homogeneity (>0.85) and low contrast (<2.0)
        if homogeneity > 0.85:
            score += min(0.35, (homogeneity - 0.85) / 0.15 * 0.35)
        
        if contrast < 1.5:
            score += (1.5 - contrast) / 1.5 * 0.25

        # Abnormally low LBP entropy indicates lack of natural high-frequency micro-texture
        if lbp_entropy < 3.2:
            score += (3.2 - lbp_entropy) / 3.2 * 0.25

        # Over-sharpened images
        if lap_var > 1500.0:
            score += min(0.15, (lap_var - 1500.0) / 2000.0 * 0.15)

        return float(np.clip(score, 0.0, 1.0))
