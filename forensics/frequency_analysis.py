import io
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image

class FrequencyAnalyzer:
    """
    Analyzes frequency domain artifacts using 2D FFT and DCT.
    Detects spectral periodicities, grid artifacts, and high-frequency discrepancies
    common to GAN upconvolutions and diffusion high-frequency denoisers.
    """

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def analyze(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Runs 2D FFT spectrum and DCT analysis."""
        # Convert to grayscale array resized for consistent frequency bins
        gray = np.array(pil_img.convert("L").resize((self.target_size, self.target_size), Image.Resampling.BILINEAR), dtype=np.float32)

        # 1. 2D Fast Fourier Transform
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)

        # Normalize magnitude spectrum for display (0 to 255)
        spec_min, spec_max = np.min(magnitude_spectrum), np.max(magnitude_spectrum)
        if spec_max > spec_min:
            norm_spectrum = ((magnitude_spectrum - spec_min) / (spec_max - spec_min) * 255).astype(np.uint8)
        else:
            norm_spectrum = np.zeros_like(magnitude_spectrum, dtype=np.uint8)

        # 2. Radial (Azimuthal) Power Profile
        radial_profile = self._azimuthal_average(magnitude_spectrum)
        
        # 3. High-Frequency vs Low-Frequency Energy Ratio
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius_low = min(h, w) // 8
        radius_high = min(h, w) // 4

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

        low_mask = dist_from_center <= radius_low
        high_mask = dist_from_center >= radius_high

        low_energy = np.sum(np.abs(f_shift)[low_mask])
        high_energy = np.sum(np.abs(f_shift)[high_mask])
        total_energy = low_energy + high_energy + 1e-9

        hf_ratio = float(high_energy / total_energy)

        # 4. Detect Periodic Grid Peaks / Spikes (GAN upsampling artifact)
        # We look at local variance in the high-frequency band of the FFT
        hf_spectrum_band = magnitude_spectrum[high_mask]
        hf_variance = float(np.var(hf_spectrum_band))
        hf_kurtosis = float(self._kurtosis(hf_spectrum_band))

        # 5. DCT 8x8 Block AC/DC distribution
        dct_metrics = self._analyze_dct(gray)

        # Synthesize Spectral Anomaly Score (0 to 1)
        # Natural images exhibit smooth 1/f^alpha radial falloff; AI generators often have spiked or suppressed high-frequencies
        spectral_anomaly_score = self._compute_anomaly_score(hf_ratio, hf_kurtosis, dct_metrics["high_freq_ac_energy"])

        return {
            "high_frequency_ratio": round(hf_ratio, 4),
            "high_frequency_variance": round(hf_variance, 4),
            "high_frequency_kurtosis": round(hf_kurtosis, 4),
            "dct_high_ac_energy": round(dct_metrics["high_freq_ac_energy"], 4),
            "dct_dc_energy_ratio": round(dct_metrics["dc_ratio"], 4),
            "spectral_anomaly_score": round(spectral_anomaly_score, 4),
            "radial_profile": [round(float(v), 2) for v in radial_profile[:64]],
            "spectrum_image": norm_spectrum
        }

    def _azimuthal_average(self, image: np.ndarray, num_bins: int = 64) -> np.ndarray:
        """Computes radially averaged power distribution around center DC component."""
        y, x = np.indices(image.shape)
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
        r = np.hypot(x - center[0], y - center[1])

        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]

        # Truncate to max radius
        max_r = min(center)
        mask = r_sorted <= max_r
        r_sorted = r_sorted[mask]
        i_sorted = i_sorted[mask]

        bin_edges = np.linspace(0, max_r, num_bins + 1)
        bin_means = []
        for i in range(num_bins):
            in_bin = (r_sorted >= bin_edges[i]) & (r_sorted < bin_edges[i+1])
            if np.any(in_bin):
                bin_means.append(np.mean(i_sorted[in_bin]))
            else:
                bin_means.append(0.0)
        return np.array(bin_means)

    def _analyze_dct(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculates 8x8 block DCT distribution."""
        h, w = gray.shape
        h_trim = (h // 8) * 8
        w_trim = (w // 8) * 8
        blocks = gray[:h_trim, :w_trim].reshape(h_trim // 8, 8, w_trim // 8, 8).transpose(0, 2, 1, 3)

        ac_energies = []
        dc_energies = []
        for i in range(min(16, blocks.shape[0])):
            for j in range(min(16, blocks.shape[1])):
                block = blocks[i, j].astype(np.float32)
                dct_block = cv2.dct(block)
                dc = np.abs(dct_block[0, 0])
                ac = np.sum(np.abs(dct_block[4:, 4:])) # high AC coefficients
                dc_energies.append(dc)
                ac_energies.append(ac)

        mean_dc = float(np.mean(dc_energies)) if dc_energies else 1.0
        mean_ac = float(np.mean(ac_energies)) if ac_energies else 0.0
        total = mean_dc + mean_ac + 1e-9

        return {
            "high_freq_ac_energy": mean_ac / total,
            "dc_ratio": mean_dc / total
        }

    def _kurtosis(self, x: np.ndarray) -> float:
        """Kurtosis calculation for tail heaviness."""
        if len(x) < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-7:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3.0)

    def _compute_anomaly_score(self, hf_ratio: float, kurt: float, dct_ac: float) -> float:
        """Heuristic calibrated anomaly score from spectral signatures."""
        score = 0.0
        # AI images frequently have either abnormally low high-freq ratio (<0.08 due to smooth denoising) or spiked high-freq (>0.35)
        if hf_ratio < 0.10:
            score += (0.10 - hf_ratio) / 0.10 * 0.4
        elif hf_ratio > 0.30:
            score += min(0.4, (hf_ratio - 0.30) / 0.20 * 0.4)

        # High kurtosis indicates spectral spike outliers
        if kurt > 3.0:
            score += min(0.3, (kurt - 3.0) / 5.0 * 0.3)

        # Low DCT AC energy indicates aggressive smoothing or generator blur
        if dct_ac < 0.05:
            score += (0.05 - dct_ac) / 0.05 * 0.3

        return float(np.clip(score, 0.0, 1.0))
