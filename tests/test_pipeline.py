import unittest
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing.image_processor import ImageProcessor
from forensics import ImageForensicSuite
from models.face_detector import FaceForensicDetector
from models.ai_detector import AIDetector
from models.filter_detector import FilterDetector
from models.ensemble import EnsembleDecisionEngine
from explainability.gradcam import GradCAMExplainer
from reports.pdf_generator import ForensicReportGenerator
from feedback.feedback_manager import FeedbackManager
from pipeline import ForensicPipeline

class TestForensicPipeline(unittest.TestCase):
    """Unit and integration test suite for the complete AI forensics system."""

    @classmethod
    def setUpClass(cls):
        # Create a synthetic 128x128 test image
        img_arr = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        cls.test_image = Image.fromarray(img_arr)
        cls.processor = ImageProcessor(target_size=128)
        cls.pipeline = ForensicPipeline()

    def test_image_processor_validation_and_hashes(self):
        valid, msg, img = self.processor.validate_image(self.test_image)
        self.assertTrue(valid)
        self.assertIsNotNone(img)

        metadata = self.processor.extract_metadata(img)
        self.assertIn("hashes", metadata)
        self.assertEqual(len(metadata["hashes"]["sha256"]), 64)
        self.assertIn("phash", metadata["hashes"])

    def test_forensics_suite(self):
        suite = ImageForensicSuite()
        results = suite.analyze_all(self.test_image)
        self.assertIn("composite_forensic_score", results)
        self.assertIn("frequency_analysis", results)
        self.assertIn("noise_analysis", results)
        self.assertIn("compression_analysis", results)
        self.assertIn("texture_analysis", results)
        self.assertTrue(0.0 <= results["composite_forensic_score"] <= 1.0)

    def test_face_detector_fallback(self):
        detector = FaceForensicDetector()
        res = detector.detect_and_analyze(self.test_image)
        self.assertIn("face_detected", res)
        self.assertIn("face_anomaly_score", res)

    def test_ai_detector(self):
        ai_det = AIDetector()
        preds = ai_det.predict(self.test_image)
        self.assertIn("predicted_class", preds)
        self.assertIn("confidence", preds)
        self.assertIn("class_probabilities", preds)
        self.assertIn("top_subfamily", preds)
        self.assertTrue(0.0 <= preds["confidence"] <= 1.0)

    def test_filter_detector(self):
        flt_det = FilterDetector()
        preds = flt_det.predict(self.test_image)
        self.assertIn("all_filter_scores", preds)
        self.assertIn("detected_filters", preds)
        self.assertEqual(len(preds["all_filter_scores"]), 10)

    def test_gradcam_explainability(self):
        ai_det = AIDetector()
        explainer = GradCAMExplainer(ai_det)
        exp_res = explainer.generate_explanation(self.test_image)
        self.assertIn("heatmap", exp_res)
        self.assertIn("overlay_image", exp_res)
        self.assertIn("forensic_reasoning", exp_res)
        self.assertEqual(exp_res["heatmap"].shape, (128, 128))

    def test_ensemble_decision_layer(self):
        ensemble = EnsembleDecisionEngine()
        res = self.pipeline.run_analysis(self.test_image, generate_cam=False)
        self.assertTrue(res["success"])
        ens = res["ensemble"]
        self.assertIn("verdict", ens)
        self.assertIn("authenticity_score", ens)
        self.assertIn("overall_confidence", ens)
        self.assertTrue(0.0 <= ens["authenticity_score"] <= 100.0)

    def test_pdf_report_generation(self):
        res = self.pipeline.run_analysis(self.test_image, generate_cam=True)
        pdf_bytes = self.pipeline.generate_pdf_report(res)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_feedback_manager(self):
        fb_mgr = FeedbackManager()
        entry_id = fb_mgr.record_feedback(
            pil_img=self.test_image,
            image_hash="abc123testsha256",
            predicted_verdict="REAL",
            predicted_ai_prob=15.0,
            predicted_confidence=85.0,
            user_label="REAL",
            user_agrees=True,
            comment="Unit test feedback verification"
        )
        self.assertIsInstance(entry_id, int)
        stats = fb_mgr.get_feedback_stats()
        self.assertGreaterEqual(stats["total_submissions"], 1)

if __name__ == "__main__":
    unittest.main()
