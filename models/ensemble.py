from typing import Dict, Any
import numpy as np

from utils.config import ENSEMBLE_WEIGHTS


class EnsembleDecisionEngine:
    """Combines classifier and forensic signals into a cautious binary verdict."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights if weights else ENSEMBLE_WEIGHTS.copy()

    def evaluate(
        self,
        ai_pred: Dict[str, Any],
        filter_pred: Dict[str, Any],
        forensics_pred: Dict[str, Any],
        face_pred: Dict[str, Any]
    ) -> Dict[str, Any]:
        class_probs = ai_pred.get("class_probabilities", {})
        ai_likeness = float(ai_pred.get("ai_likeness_score", 0.5))
        raw_ai_prob = float(class_probs.get("AI_GENERATED", 0.0))
        raw_edit_prob = float(class_probs.get("AI_EDITED", 0.0))
        raw_filter_prob = float(class_probs.get("FILTERED", 0.0))
        raw_manip_prob = float(class_probs.get("MANIPULATED", 0.0))

        freq_anomaly = float(forensics_pred.get("frequency_analysis", {}).get("spectral_anomaly_score", 0.0))
        forensic_composite = float(forensics_pred.get("composite_forensic_score", 0.0))
        has_face = bool(face_pred.get("face_detected", False))
        face_anomaly = float(face_pred.get("face_anomaly_score", 0.0)) if has_face else 0.0

        w_deep = self.weights.get("deep_ai_model", 0.45)
        w_freq = self.weights.get("frequency_analysis", 0.20)
        w_forensic = self.weights.get("image_forensics", 0.20)
        w_face = self.weights.get("face_analysis", 0.15)
        if not has_face:
            total_remaining = w_deep + w_freq + w_forensic
            if total_remaining > 0:
                w_deep /= total_remaining
                w_freq /= total_remaining
                w_forensic /= total_remaining
            w_face = 0.0

        synthetic_index = float(np.clip(
            w_deep * ai_likeness + w_freq * freq_anomaly +
            w_forensic * forensic_composite + w_face * face_anomaly,
            0.0, 1.0
        ))

        # Combine available evidence, then normalize all five categories.
        real_evidence = max(0.0, 1.0 - synthetic_index)
        ai_evidence = max(0.0, 0.6 * raw_ai_prob + 0.4 * synthetic_index)
        filter_evidence = max(0.0, 0.5 * raw_filter_prob + 0.5 * float(filter_pred.get("max_filter_score", 0.0)))
        manipulation_evidence = max(0.0, 0.5 * raw_manip_prob + 0.5 * forensic_composite)
        edit_evidence = max(0.0, raw_edit_prob)
        evidence = np.array([real_evidence, ai_evidence, edit_evidence, filter_evidence, manipulation_evidence], dtype=float)
        total = float(evidence.sum())
        normalized = evidence / total if total > 0 else np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        norm_real, norm_ai, norm_edit, norm_filter, norm_manip = (float(x) for x in normalized)

        # A filtered or manipulated image is not an untouched authentic capture.
        fake_probability = norm_ai + norm_edit + norm_filter + norm_manip
        binary_confidence = max(norm_real, fake_probability)
        # Never present an untrained/random classifier or a weakly separated result as fact.
        model_trained = bool(ai_pred.get("model_trained", True))
        is_uncertain = (
            not model_trained or binary_confidence < 0.65 or
            abs(norm_real - fake_probability) < 0.20
        )

        # Keep the more specific category for diagnostics, while exposing one clear verdict.
        if norm_ai >= max(norm_real, norm_edit, norm_filter, norm_manip):
            category_verdict = "AI-GENERATED"
            category_summary = "Signals indicate the image may be AI-generated."
        elif norm_edit >= max(norm_real, norm_ai, norm_filter, norm_manip):
            category_verdict = "AI-EDITED / INPAINTED"
            category_summary = "Signals indicate localized AI editing or generative fill."
        elif norm_manip >= max(norm_real, norm_ai, norm_filter):
            category_verdict = "MANIPULATED / PHOTOSHOPPED"
            category_summary = "Signals indicate possible digital manipulation."
        elif norm_filter >= norm_real:
            category_verdict = "HEAVILY FILTERED"
            category_summary = "Signals indicate significant post-processing or filtering."
        else:
            category_verdict = "REAL / NATURAL"
            category_summary = "Available signals lean toward a natural, unaltered image."

        if is_uncertain:
            verdict = "UNCERTAIN"
            verdict_badge = "⚠️ UNCERTAIN — cannot reliably determine"
            verdict_color = "#f59e0b"
            if not model_trained:
                summary = "No trained authenticity checkpoint is loaded. This result is only a heuristic and cannot reliably determine real vs. fake."
            else:
                summary = "The available evidence is not strong or consistent enough to label this image real or fake reliably."
        elif norm_real > fake_probability:
            verdict = "REAL"
            verdict_badge = "🌿 LIKELY REAL"
            verdict_color = "#10b981"
            summary = "The available evidence favors a real image; automated analysis cannot guarantee authenticity."
        else:
            verdict = "FAKE / ALTERED"
            verdict_badge = "⚠️ LIKELY FAKE / ALTERED"
            verdict_color = "#ef4444"
            summary = f"The available evidence favors a synthetic or altered image ({category_verdict.lower()}); automated analysis cannot guarantee authenticity."

        probabilities = {
            "real": round(norm_real * 100.0, 1),
            "ai_generated": round(norm_ai * 100.0, 1),
            "ai_edited": round(norm_edit * 100.0, 1),
            "filtered": round(norm_filter * 100.0, 1),
            "manipulated": round(norm_manip * 100.0, 1),
            "fake_or_altered": round(fake_probability * 100.0, 1)
        }
        return {
            "verdict": verdict,
            "binary_verdict": verdict,
            "category_verdict": category_verdict,
            "verdict_badge": verdict_badge,
            "verdict_color": verdict_color,
            "summary": summary,
            "category_summary": category_summary,
            "authenticity_score": round(norm_real * 100.0, 1),
            "overall_confidence": round(binary_confidence * 100.0, 1),
            "is_uncertain": is_uncertain,
            "model_trained": model_trained,
            "probabilities": probabilities,
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
