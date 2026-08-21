from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image

class NoiseAnalyzer:
    """
    Analyzes high-frequency residual noise patterns and local variance consistency.
    Physical camera sensors generate uniform Poisson-Gaussian photon shot noise.
    AI inpainting, splicing, or diffusion denoising creates noise inconsistency across regions.
    """

    def __init__(self, block_size: int = 32):
        self.block_size = block_size

    def analyze(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Calculates noise residuals, block variance maps, and inconsistency scores."""
        # Convert to float grayscale
        gray = np.array(pil_img.convert("L"), dtype=np.float32)

        # 1. Extract Noise Residual via Median Filter subtraction
        # Residual = Image - MedianFiltered(Image)
        denoised = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
        residual = gray - denoised

        # Global noise standard deviation
        global_noise_std = float(np.std(residual))
        global_noise_mean = float(np.mean(np.abs(residual)))

        # 2. Block-wise Noise Inconsistency Map
        h, w = gray.shape
        bh = h // self.block_size
        bw = w // self.block_size

        if bh < 2 or bw < 2:
            # Fallback for very small images
            block_vars = np.array([[global_noise_std]])
        else:
            block_vars = []
            for i in range(bh):
                row_vars = []
                for j in range(bw):
                    block_res = residual[
                        i * self.block_size : (i + 1) * self.block_size,
                        j * self.block_size : (j + 1) * self.block_size
                    ]
                    row_vars.append(np.var(block_res))
                block_vars.append(row_vars)
            block_vars = np.array(block_vars)

        # Variance of block noise variances (high value = spliced or locally manipulated)
        inter_block_var = float(np.var(block_vars)) if block_vars.size > 0 else 0.0
        normalized_inconsistency = float(np.std(block_vars) / (np.mean(block_vars) + 1e-6)) if block_vars.size > 0 else 0.0

        # 3. High-Pass Filter Noise Residual (Laplacian residual)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian_var = float(np.var(laplacian))

        # 4. Generate Visual Noise Map for UI display
        # Scale residual to 0-255 range
        norm_noise_map = cv2.normalize(np.abs(residual), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_noise = cv2.applyColorMap(norm_noise_map, cv2.COLORMAP_JET)
        heatmap_noise_rgb = cv2.cvtColor(heatmap_noise, cv2.COLOR_BGR2RGB)

        # Anomaly score calculation
        anomaly_score = self._compute_noise_anomaly(global_noise_std, normalized_inconsistency, laplacian_var)

        return {
            "global_noise_std": round(global_noise_std, 4),
            "global_noise_mean": round(global_noise_mean, 4),
            "noise_inconsistency_score": round(anomaly_score, 4),
            "normalized_inconsistency": round(normalized_inconsistency, 4),
            "laplacian_noise_var": round(laplacian_var, 4),
            "noise_map_rgb": heatmap_noise_rgb
        }

    def _compute_noise_anomaly(self, noise_std: float, inconsistency: float, lap_var: float) -> float:
        """Calibrates noise metrics into a 0.0 - 1.0 tampering indicator score."""
        score = 0.0
        # AI images typically have unnaturally low noise std (<1.5) or synthetic high inconsistency (>1.2)
        if noise_std < 1.8:
            score += (1.8 - noise_std) / 1.8 * 0.4
        elif noise_std > 15.0:
            score += min(0.3, (noise_std - 15.0) / 20.0 * 0.3)

        if inconsistency > 0.8:
            score += min(0.4, (inconsistency - 0.8) / 1.2 * 0.4)

        if lap_var < 50.0:
            score += (50.0 - lap_var) / 50.0 * 0.2

        return float(np.clip(score, 0.0, 1.0))
