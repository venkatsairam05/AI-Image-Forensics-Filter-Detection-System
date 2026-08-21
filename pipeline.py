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
from models.ai_detector import AIDetector
from models.filter_detector import FilterDetector
from models.ensemble import EnsembleDecisionEngine
from explainability.gradcam import GradCAMExplainer
from reports.pdf_generator import ForensicReportGenerator

class ForensicPipeline:
    """
    Production end-to-end multimodal AI Image and Filter Forensics Pipeline.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        logger.info(f"Initializing ForensicPipeline on {device}...")
        self.processor = ImageProcessor()
        self.forensic_suite = ImageForensicSuite()
        self.face_detector = FaceForensicDetector()
        self.ai_detector = AIDetector(device=device)
        self.filter_detector = FilterDetector(device=device)
        self.ensemble = EnsembleDecisionEngine()
        self.explainer = GradCAMExplainer(self.ai_detector)
        self.pdf_generator = ForensicReportGenerator()
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
        ai_results = self.ai_detector.predict(pil_img)

        # 6. Multi-Label Filter Detection (fused with forensic signals)
        filter_results = self.filter_detector.predict(pil_img, forensic_signals={
            "texture_analysis": forensics_results.get("texture_analysis", {}),
            "face_analysis": face_results,
            "compression_analysis": forensics_results.get("compression_analysis", {})
        })

        # 7. Multimodal Ensemble Decision
        ensemble_results = self.ensemble.evaluate(
            ai_pred=ai_results,
            filter_pred=filter_results,
            forensics_pred=forensics_results,
            face_pred=face_results
        )

        # 8. Explainable AI (Grad-CAM)
        explainability_results = {}
        if generate_cam:
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
                    "forensic_reasoning": "Explainability visualizer encountered an error during backpropagation.",
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
