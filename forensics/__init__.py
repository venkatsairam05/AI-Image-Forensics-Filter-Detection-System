# forensics package
import numpy as np
from .frequency_analysis import FrequencyAnalyzer
from .noise_analysis import NoiseAnalyzer
from .compression_analysis import CompressionAnalyzer
from .texture_analysis import TextureAnalyzer

class ImageForensicSuite:
    """Unified forensics engine coordinating frequency, noise, compression, and texture analyses."""

    def __init__(self):
        self.freq = FrequencyAnalyzer()
        self.noise = NoiseAnalyzer()
        self.compression = CompressionAnalyzer()
        self.texture = TextureAnalyzer()

    def analyze_all(self, pil_img):
        """Runs full suite of computer vision forensics and returns aggregated metrics."""
        freq_results = self.freq.analyze(pil_img)
        noise_results = self.noise.analyze(pil_img)
        comp_results = self.compression.analyze(pil_img)
        text_results = self.texture.analyze(pil_img)

        # Calculate a normalized forensic manipulation score (0.0 to 1.0)
        # Higher score = stronger indicator of synthetic or manipulated tampering
        tampering_signals = [
            freq_results.get("spectral_anomaly_score", 0.0),
            noise_results.get("noise_inconsistency_score", 0.0),
            comp_results.get("ela_anomaly_score", 0.0),
            text_results.get("texture_anomaly_score", 0.0)
        ]
        composite_forensic_score = float(np.mean(tampering_signals)) if tampering_signals else 0.0

        return {
            "composite_forensic_score": round(composite_forensic_score, 4),
            "frequency_analysis": freq_results,
            "noise_analysis": noise_results,
            "compression_analysis": comp_results,
            "texture_analysis": text_results
        }
