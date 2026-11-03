import unittest
import sys
from pathlib import Path
import numpy as np
import torch
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
from utils.config import PRIMARY_CLASSES, AI_SUBFAMILIES


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


class TestAIDetectorDecisionRules(unittest.TestCase):
    """Regression tests for the REAL-versus-synthetic inference policy."""

    @staticmethod
    def _predict_with_probabilities(probabilities):
        detector = AIDetector.__new__(AIDetector)
        detector.device = "cpu"

        class StubProcessor:
            @staticmethod
            def to_tensor(_image, device=None):
                return torch.zeros((1, 3, 8, 8), device=device)

        class StubModel:
            def __call__(self, _tensor):
                # Softmax(log(probabilities)) reproduces the specified test probabilities.
                primary_logits = torch.tensor(
                    [np.log([probabilities[label] for label in PRIMARY_CLASSES])],
                    dtype=torch.float32,
                )
                subfamily_logits = torch.zeros((1, len(AI_SUBFAMILIES)), dtype=torch.float32)
                return primary_logits, subfamily_logits

        detector.processor = StubProcessor()
        detector.model = StubModel()
        return detector.predict(Image.new("RGB", (8, 8)))

    def test_synthetic_signal_is_not_overridden_by_slightly_higher_real_score(self):
        result = self._predict_with_probabilities({
            "REAL": 0.50,
            "AI_GENERATED": 0.40,
            "AI_EDITED": 0.04,
            "FILTERED": 0.03,
            "MANIPULATED": 0.03,
        })
        self.assertEqual(result["predicted_class"], "AI_GENERATED")
        self.assertGreater(result["ai_likeness_score"], 0.20)

    def test_clear_real_signal_remains_real(self):
        result = self._predict_with_probabilities({
            "REAL": 0.85,
            "AI_GENERATED": 0.04,
            "AI_EDITED": 0.03,
            "FILTERED": 0.05,
            "MANIPULATED": 0.03,
        })
        self.assertEqual(result["predicted_class"], "REAL")
        self.assertEqual(result["top_subfamily"], "Natural Photographic Capture")


if __name__ == "__main__":
    unittest.main()
