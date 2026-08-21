import io
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

class CompressionAnalyzer:
    """
    Forensic Error Level Analysis (ELA) and JPEG block grid artifact analyzer.
    Identifies differing compression rates across image regions (e.g. pasted or in-painted elements).
    """

    def __init__(self, ela_quality: int = 90, scale_factor: int = 15):
        self.ela_quality = ela_quality
        self.scale_factor = scale_factor

    def analyze(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Performs ELA and 8x8 JPEG grid boundary analysis."""
        rgb_img = pil_img.convert("RGB")
        
        # 1. Error Level Analysis (ELA)
        buffer = io.BytesIO()
        rgb_img.save(buffer, 'JPEG', quality=self.ela_quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        # Compute absolute difference
        diff = ImageChops.difference(rgb_img, recompressed)
        
        # Calculate numerical statistics before amplifying
        diff_arr = np.array(diff, dtype=np.float32)
        mean_ela = float(np.mean(diff_arr))
        max_ela = float(np.max(diff_arr))
        std_ela = float(np.std(diff_arr))

        # Amplify visual difference
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale = 255.0 / max(1, max_diff) if max_diff > 0 else 1.0
        # Use configurable scale factor or auto-scale
        visual_scale = min(scale, self.scale_factor)
        ela_visual = ImageEnhance.Brightness(diff).enhance(visual_scale)
        ela_visual_np = np.array(ela_visual)

        # 2. JPEG 8x8 Grid Artifacts Detection
        grid_strength = self._detect_jpeg_grid(rgb_img)

        # 3. Anomaly Score
        ela_anomaly_score = self._compute_ela_anomaly(mean_ela, std_ela, grid_strength)

        return {
            "mean_ela_error": round(mean_ela, 4),
            "max_ela_error": round(max_ela, 4),
            "std_ela_error": round(std_ela, 4),
            "jpeg_grid_strength": round(grid_strength, 4),
            "ela_anomaly_score": round(ela_anomaly_score, 4),
            "ela_image": ela_visual_np
        }

    def _detect_jpeg_grid(self, rgb_img: Image.Image) -> float:
        """Measures periodic differences along 8x8 JPEG macroblock boundaries."""
        gray = np.array(rgb_img.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.0

        # Calculate horizontal and vertical differences
        dh = np.abs(gray[1:, :] - gray[:-1, :])
        dw = np.abs(gray[:, 1:] - gray[:, :-1])

        # Grid positions (every 8th pixel)
        grid_rows = [i for i in range(7, dh.shape[0], 8)]
        grid_cols = [j for j in range(7, dw.shape[1], 8)]
        non_grid_rows = [i for i in range(dh.shape[0]) if i % 8 != 7]
        non_grid_cols = [j for j in range(dw.shape[1]) if j % 8 != 7]

        grid_diff_h = np.mean(dh[grid_rows, :]) if grid_rows else 0.0
        non_grid_diff_h = np.mean(dh[non_grid_rows, :]) if non_grid_rows else 1.0

        grid_diff_w = np.mean(dw[:, grid_cols]) if grid_cols else 0.0
        non_grid_diff_w = np.mean(dw[:, non_grid_cols]) if non_grid_cols else 1.0

        ratio_h = grid_diff_h / (non_grid_diff_h + 1e-6)
        ratio_w = grid_diff_w / (non_grid_diff_w + 1e-6)

        grid_strength = float((ratio_h + ratio_w) / 2.0 - 1.0)
        return float(np.clip(grid_strength, 0.0, 5.0))

    def _compute_ela_anomaly(self, mean_ela: float, std_ela: float, grid_strength: float) -> float:
        """Calibrates ELA and compression signals into an anomaly probability."""
        score = 0.0
        # Extreme ELA standard deviation or mean indicates inconsistent compression levels
        if std_ela > 6.0:
            score += min(0.45, (std_ela - 6.0) / 10.0 * 0.45)
        elif std_ela < 1.0:
            # Uncompressed or direct synthetic export often has extremely uniform low ELA
            score += (1.0 - std_ela) / 1.0 * 0.25

        if grid_strength > 0.35:
            score += min(0.3, (grid_strength - 0.35) / 1.0 * 0.3)

        return float(np.clip(score, 0.0, 1.0))
