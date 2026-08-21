from typing import Dict, Any, List
import numpy as np

from utils.config import (
    ENSEMBLE_WEIGHTS, UNCERTAINTY_THRESHOLD_LOW, UNCERTAINTY_THRESHOLD_HIGH
)

class EnsembleDecisionEngine:
    """
    Multimodal Forensic Ensemble Layer.
    Fuses deep neural probabilities, multi-label filter detections,
    frequency spectrum anomalies, noise inconsistency, ELA compression,
    and facial forensics into a calibrated authenticity verdict.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights if weights else ENSEMBLE_WEIGHTS.copy()

    def evaluate(
        self,
        ai_pred: Dict[str, Any],
        filter_pred: Dict[str, Any],
        forensics_pred: Dict[str, Any],
        face_pred: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combines multi-stream forensic signals to generate the final authenticity verdict and probabilities.
        """
        # 1. Extract Individual Signals
        ai_likeness = ai_pred.get("ai_likeness_score", 0.5)
        raw_real_prob = ai_pred.get("class_probabilities", {}).get("REAL", 0.5)
        raw_ai_prob = ai_pred.get("class_probabilities", {}).get("AI_GENERATED", 0.5)
        raw_edit_prob = ai_pred.get("class_probabilities", {}).get("AI_EDITED", 0.0)
        raw_filt_prob = ai_pred.get("class_probabilities", {}).get("FILTERED", 0.0)
        raw_manip_prob = ai_pred.get("class_probabilities", {}).get("MANIPULATED", 0.0)

        freq_anomaly = forensics_pred.get("frequency_analysis", {}).get("spectral_anomaly_score", 0.0)
        forensic_composite = forensics_pred.get("composite_forensic_score", 0.0)
        
        has_face = face_pred.get("face_detected", False)
        face_anomaly = face_pred.get("face_anomaly_score", 0.0) if has_face else 0.0

        # 2. Dynamic Weight Rebalancing (if no face is detected, redistribute weight)
        w_deep = self.weights.get("deep_ai_model", 0.45)
        w_freq = self.weights.get("frequency_analysis", 0.20)
        w_forensic = self.weights.get("image_forensics", 0.20)
        w_face = self.weights.get("face_analysis", 0.15)

        if not has_face:
            total_remaining = w_deep + w_freq + w_forensic
            w_deep = (w_deep / total_remaining) * 1.0
            w_freq = (w_freq / total_remaining) * 1.0
            w_forensic = (w_forensic / total_remaining) * 1.0
            w_face = 0.0

        # 3. Compute Composite Synthetic / Tampering Index
        synthetic_index = (
            w_deep * ai_likeness +
            w_freq * freq_anomaly +
            w_forensic * forensic_composite +
            w_face * face_anomaly
        )
        synthetic_index = float(np.clip(synthetic_index, 0.0, 1.0))

        # 4. Calibrated Ensemble Probabilities
        final_ai_prob = float(np.clip(0.6 * raw_ai_prob + 0.4 * synthetic_index, 0.0, 1.0))
        final_real_prob = float(np.clip(1.0 - synthetic_index, 0.0, 1.0))
        
        # Max filter score influence
        max_filter_score = filter_pred.get("max_filter_score", 0.0)
        final_filter_prob = float(np.clip(0.5 * raw_filt_prob + 0.5 * max_filter_score, 0.0, 1.0))
        final_manip_prob = float(np.clip(0.5 * raw_manip_prob + 0.5 * forensic_composite, 0.0, 1.0))

        # Normalize probability distribution
        total_p = final_real_prob + final_ai_prob + raw_edit_prob + final_filter_prob + final_manip_prob
        if total_p > 0:
            norm_real = final_real_prob / total_p
            norm_ai = final_ai_prob / total_p
            norm_edit = raw_edit_prob / total_p
            norm_filt = final_filter_prob / total_p
            norm_manip = final_manip_prob / total_p
        else:
            norm_real, norm_ai, norm_edit, norm_filt, norm_manip = 0.5, 0.5, 0.0, 0.0, 0.0

        # 5. Authenticity Score (100 = Authentic Real Natural, 0 = High Synthetic / Tampered)
        authenticity_score = round(float(norm_real * 100.0), 1)

        # 6. Overall Confidence Calculation
        top_prob = max(norm_real, norm_ai, norm_edit, norm_filt, norm_manip)
        confidence = float(np.clip(top_prob, 0.50, 0.99))

        # 7. Uncertainty Assessment
        # If margins between Real and Synthetic are too narrow, flag as UNCERTAIN
        is_uncertain = False
        if abs(norm_real - (norm_ai + norm_edit + norm_manip)) < 0.15 and (0.35 <= norm_real <= 0.65):
            is_uncertain = True

        # 8. Verdict Determination
        if is_uncertain:
            verdict = "UNCERTAIN / AMBIGUOUS"
            verdict_badge = "⚠️ Uncertain"
            verdict_color = "#f59e0b" # amber
            summary = "The forensic signals are inconclusive. The image contains subtle mixed indicators that cannot definitively separate natural optical capture from synthetic processing."
        elif norm_ai >= max(norm_real, norm_edit, norm_filt, norm_manip):
            verdict = "AI-GENERATED"
            verdict_badge = "🤖 AI-Generated"
            verdict_color = "#ef4444" # red
            summary = f"High synthetic probability detected. Image exhibits characteristic generative patterns ({ai_pred.get('top_subfamily', 'Synthetic')}) and spectral frequency artifacts."
        elif norm_edit >= max(norm_real, norm_ai, norm_filt, norm_manip):
            verdict = "AI-EDITED / INPAINTED"
            verdict_badge = "🎨 AI-Edited"
            verdict_color = "#ec4899" # pink
            summary = "Localized AI editing or generative fill detected. Regions show noise and compression boundary inconsistencies."
        elif norm_manip >= max(norm_real, norm_ai, norm_filt):
            verdict = "MANIPULATED / PHOTOSHOPPED"
            verdict_badge = "✂️ Manipulated"
            verdict_color = "#8b5cf6" # purple
            summary = "Digital manipulation detected. ELA compression and edge gradient forensics indicate composite tampering."
        elif norm_filt >= norm_real:
            verdict = "HEAVILY FILTERED"
            verdict_badge = "✨ Heavily Filtered"
            verdict_color = "#3b82f6" # blue
            summary = f"Significant post-processing filters detected ({', '.join(filter_pred.get('detected_filter_names', ['filters'])[:3])})."
        else:
            verdict = "REAL / NATURAL"
            verdict_badge = "🌿 Real / Natural"
            verdict_color = "#10b981" # green
            summary = "Consistent natural sensor noise, organic optical frequency falloff, and uniform compression without synthetic artifacts."

        return {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "verdict_color": verdict_color,
            "summary": summary,
            "authenticity_score": authenticity_score,
            "overall_confidence": round(confidence * 100.0, 1),
            "is_uncertain": is_uncertain,
            "probabilities": {
                "real": round(norm_real * 100.0, 1),
                "ai_generated": round(norm_ai * 100.0, 1),
                "ai_edited": round(norm_edit * 100.0, 1),
                "filtered": round(norm_filt * 100.0, 1),
                "manipulated": round(norm_manip * 100.0, 1)
            },
            "weights_used": {
                "deep_ai_model": round(w_deep, 2),
                "frequency_analysis": round(w_freq, 2),
                "image_forensics": round(w_forensic, 2),
                "face_analysis": round(w_face, 2)
            },
            "signal_breakdown": {
                "deep_ai_likeness": round(ai_likeness, 4),
                "spectral_anomaly": round(freq_anomaly, 4),
                "forensic_composite": round(forensic_composite, 4),
                "face_anomaly": round(face_anomaly, 4)
            }
        }
