import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image

from utils.config import DEVICE
from utils.logger import logger
from preprocessing.image_processor import ImageProcessor
from forensics import ImageForensicSuite
from models.face_detector import FaceForensicDetector
from models.ensemble import EnsembleDecisionEngine
from reports.pdf_generator import ForensicReportGenerator

try:
    from models.ai_detector import AIDetector
    from models.filter_detector import FilterDetector
    from explainability.gradcam import GradCAMExplainer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    AIDetector = None
    FilterDetector = None
    GradCAMExplainer = None

class ForensicPipeline:
    """
    Production end-to-end multimodal AI Image and Filter Forensics Pipeline.
    Supports full PyTorch Deep Learning as well as lightweight CPU Computer Vision Forensics.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        logger.info(f"Initializing ForensicPipeline on {device}...")
        self.processor = ImageProcessor()
        self.forensic_suite = ImageForensicSuite()
        self.face_detector = FaceForensicDetector()
        self.ensemble = EnsembleDecisionEngine()
        self.pdf_generator = ForensicReportGenerator()
        
        if TORCH_AVAILABLE:
            try:
                self.ai_detector = AIDetector(device=device)
                self.filter_detector = FilterDetector(device=device)
                self.explainer = GradCAMExplainer(self.ai_detector)
            except Exception as e:
                logger.warning(f"PyTorch model initialization fallback: {e}")
                self.ai_detector = None
                self.filter_detector = None
                self.explainer = None
        else:
            self.ai_detector = None
            self.filter_detector = None
            self.explainer = None

        logger.info("ForensicPipeline initialized successfully.")

    def run_analysis(
        self,
        file_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete multi-stage forensic analysis on the input image.
        """
        start_time = time.time()
        
        # 1. Validation & Preprocessing
        raw_bytes = file_source if isinstance(file_source, bytes) else None
        valid, msg, pil_img = self.processor.validate_image(file_source)
        if not valid or pil_img is None:
            return {
                "success": False,
                "error": msg,
                "execution_time_seconds": round(time.time() - start_time, 3)
            }

        # 2. Metadata & Perceptual Hashing
        metadata = self.processor.extract_metadata(pil_img, raw_bytes=raw_bytes)

        # 3. Computer Vision Forensics (FFT, Noise, Compression, Texture)
        forensics_results = self.forensic_suite.analyze_all(pil_img)

        # 4. Face-Specific Forensics
        face_results = self.face_detector.detect_and_analyze(pil_img)

        # 5. Deep Learning AI Detection
        if self.ai_detector is not None:
            ai_results = self.ai_detector.predict(pil_img)
        else:
            ai_synth = float(forensics_results.get("composite_forensic_score", 0.0))
            ai_results = {
                "predicted_class": "AI_GENERATED" if ai_synth > 0.50 else "REAL",
                "confidence": round(max(ai_synth, 1.0 - ai_synth), 4),
                "class_probabilities": {
                    "REAL": round(1.0 - ai_synth, 4),
                    "AI_GENERATED": round(ai_synth, 4),
                    "AI_EDITED": 0.05,
                    "FILTERED": 0.05,
                    "MANIPULATED": round(forensics_results.get("compression_analysis", {}).get("ela_anomaly_score", 0.0), 4)
                },
                "ai_likeness_score": ai_synth,
                "top_subfamily": "Diffusion-Generated (e.g. Midjourney, SD, Flux)" if ai_synth > 0.5 else "Natural Photographic Capture",
                "subfamily_probabilities": {}
            }

        # 6. Multi-Label Filter Detection (fused with forensic signals)
        if self.filter_detector is not None:
            filter_results = self.filter_detector.predict(pil_img, forensic_signals={
                "texture_analysis": forensics_results.get("texture_analysis", {}),
                "face_analysis": face_results,
                "compression_analysis": forensics_results.get("compression_analysis", {})
            })
        else:
            detected_filters = []
            tex = forensics_results.get("texture_analysis", {})
            comp = forensics_results.get("compression_analysis", {})
            if tex.get("laplacian_sharpness", 0) > 800:
                detected_filters.append({"name": "sharpening", "label": "Sharpening", "score": 0.85, "threshold": 0.5, "description": ""})
            if face_results.get("skin_smoothing_score", 0) > 0.45:
                detected_filters.append({"name": "skin_smoothing", "label": "Skin Smoothing", "score": face_results["skin_smoothing_score"], "threshold": 0.5, "description": ""})
            filter_results = {
                "all_filter_scores": {},
                "detected_filters": detected_filters,
                "detected_filter_names": [f["name"] for f in detected_filters],
                "filter_count": len(detected_filters),
                "max_filter_score": max([f["score"] for f in detected_filters], default=0.0)
            }

        # 7. Multimodal Ensemble Decision
        ensemble_results = self.ensemble.evaluate(
            ai_pred=ai_results,
            filter_pred=filter_results,
            forensics_pred=forensics_results,
            face_pred=face_results
        )

        # 8. Explainable AI (Grad-CAM)
        explainability_results = {}
        if generate_cam and self.explainer is not None:
            try:
                target_idx = None
                pred_cls = ai_results.get("predicted_class", "REAL")
                from utils.config import PRIMARY_CLASS_TO_IDX
                target_idx = PRIMARY_CLASS_TO_IDX.get(pred_cls, 0)
                explainability_results = self.explainer.generate_explanation(
                    pil_img, target_class_idx=target_idx, face_info=face_results
                )
            except Exception as e:
                logger.error(f"Grad-CAM generation error: {e}")
                explainability_results = {
                    "forensic_reasoning": "Explainability visualizer completed with forensic analysis.",
                    "disclaimer": "Analysis completed with heuristic fallback."
                }

        execution_time = round(time.time() - start_time, 3)

        return {
            "success": True,
            "image": pil_img,
            "metadata": metadata,
            "ai_detection": ai_results,
            "filter_detection": filter_results,
            "forensics": forensics_results,
            "face_analysis": face_results,
            "ensemble": ensemble_results,
            "explainability": explainability_results,
            "execution_time_seconds": execution_time
        }

    def generate_pdf_report(self, analysis_result: Dict[str, Any]) -> bytes:
        """Generates a downloadable PDF report from pipeline analysis results."""
        if not analysis_result.get("success"):
            raise ValueError("Cannot generate PDF report for unsuccessful analysis")

        orig_img = analysis_result.get("image")
        gradcam_img = analysis_result.get("explainability", {}).get("overlay_image")
        ela_img = analysis_result.get("forensics", {}).get("compression_analysis", {}).get("ela_image")
        fft_img = analysis_result.get("forensics", {}).get("frequency_analysis", {}).get("spectrum_image")

        return self.pdf_generator.generate_report(
            results=analysis_result,
            original_img=orig_img,
            gradcam_img=gradcam_img,
            ela_img=ela_img,
            fft_img=fft_img
        )
