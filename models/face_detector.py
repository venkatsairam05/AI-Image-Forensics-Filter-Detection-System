import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

class FaceForensicDetector:
    """
    Facial detection and deep-fake / beauty-filter forensic analyzer.
    Detects faces, analyzes facial skin texture, unnatural smoothing,
    eye/hair boundary blending artifacts, and facial feature symmetry.
    """

    def __init__(self):
        # Load OpenCV Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if os.path.exists(eye_cascade_path):
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        else:
            self.eye_cascade = None

    def detect_and_analyze(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Executes face detection and forensic facial analysis.
        Returns face bounding boxes, annotated image, and facial manipulation metrics.
        """
        rgb_arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        faces = []
        if self.face_cascade is not None:
            detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )
            for (x, y, fw, fh) in detected:
                faces.append((int(x), int(y), int(fw), int(fh)))

        if not faces:
            return {
                "face_detected": False,
                "face_count": 0,
                "face_boxes": [],
                "annotated_image": rgb_arr,
                "skin_smoothing_score": 0.0,
                "boundary_inconsistency_score": 0.0,
                "face_anomaly_score": 0.0,
                "facial_metrics": {}
            }

        # Select primary (largest) face
        faces.sort(key=lambda b: b[2] * b[3], reverse=True)
        primary_box = faces[0]
        px, py, pw, ph = primary_box

        # Crop face ROI
        face_roi = rgb_arr[py : py + ph, px : px + pw]
        face_gray = gray[py : py + ph, px : px + pw]

        # 1. Skin Smoothing Analysis in Facial Region
        skin_smoothing_score = self._analyze_skin_smoothing(face_roi, face_gray)

        # 2. Eye & Boundary Blending Inconsistency
        boundary_inconsistency = self._analyze_face_boundary(gray, primary_box)

        # 3. Facial Feature Symmetry & Proportions
        eye_count = 0
        if self.eye_cascade is not None:
            eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            eye_count = len(eyes)

        # 4. Generate Annotated Bounding Box Image
        annotated = rgb_arr.copy()
        for i, (x, y, fw, fh) in enumerate(faces):
            color = (0, 255, 128) if i == 0 else (255, 165, 0)
            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(
                annotated,
                f"Face #{i+1} [Smoothing: {skin_smoothing_score:.2f}]",
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        # Overall facial anomaly score
        face_anomaly_score = float(np.clip(0.6 * skin_smoothing_score + 0.4 * boundary_inconsistency, 0.0, 1.0))

        return {
            "face_detected": True,
            "face_count": len(faces),
            "face_boxes": faces,
            "primary_box": list(primary_box),
            "annotated_image": annotated,
            "skin_smoothing_score": round(skin_smoothing_score, 4),
            "boundary_inconsistency_score": round(boundary_inconsistency, 4),
            "face_anomaly_score": round(face_anomaly_score, 4),
            "facial_metrics": {
                "detected_eyes": eye_count,
                "face_area_ratio": round((pw * ph) / (w * h), 4),
                "primary_face_width": pw,
                "primary_face_height": ph
            }
        }

    def _analyze_skin_smoothing(self, face_rgb: np.ndarray, face_gray: np.ndarray) -> float:
        """
        Estimates pore & skin texture degradation.
        Natural skin exhibits high-frequency micro-gradients; beauty filters / AI smooth these away.
        """
        if face_gray.size < 100:
            return 0.0

        # Segment skin tones using YCrCb color space
        ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

        if np.sum(skin_mask) < 50:
            # Fallback if color mask missed
            skin_pixels = face_gray
        else:
            skin_pixels = face_gray[skin_mask]

        # Calculate high-frequency texture gradient in skin area
        laplacian = cv2.Laplacian(face_gray, cv2.CV_32F)
        if np.sum(skin_mask) > 50:
            skin_lap = laplacian[skin_mask]
        else:
            skin_lap = laplacian

        skin_lap_var = float(np.var(skin_lap))

        # Bilateral filter comparison
        bilateral = cv2.bilateralFilter(face_gray, d=7, sigmaColor=50, sigmaSpace=50)
        diff = np.abs(face_gray.astype(np.float32) - bilateral.astype(np.float32))
        diff_mean = float(np.mean(diff))

        # If diff_mean is tiny (<1.2) and laplacian variance is very low (<100), skin is heavily smoothed
        smoothing_score = 0.0
        if diff_mean < 2.5:
            smoothing_score += (2.5 - diff_mean) / 2.5 * 0.55
        if skin_lap_var < 150.0:
            smoothing_score += (150.0 - skin_lap_var) / 150.0 * 0.45

        return float(np.clip(smoothing_score, 0.0, 1.0))

    def _analyze_face_boundary(self, gray: np.ndarray, box: Tuple[int, int, int, int]) -> float:
        """
        Detects gradient discontinuities or blur mismatches between face ROI and surrounding background.
        DeepFake face-swaps often have blending artifacts at the outer boundary.
        """
        h, w = gray.shape
        x, y, fw, fh = box
        
        # Inner face gradient
        inner_roi = gray[y + int(fh*0.15) : y + int(fh*0.85), x + int(fw*0.15) : x + int(fw*0.85)]
        inner_var = float(cv2.Laplacian(inner_roi, cv2.CV_32F).var()) if inner_roi.size > 0 else 100.0

        # Outer ring around face boundary
        bx1, by1 = max(0, x - 10), max(0, y - 10)
        bx2, by2 = min(w, x + fw + 10), min(h, y + fh + 10)
        outer_roi = gray[by1:by2, bx1:bx2]
        outer_var = float(cv2.Laplacian(outer_roi, cv2.CV_32F).var()) if outer_roi.size > 0 else 100.0

        # Discrepancy ratio
        ratio = inner_var / (outer_var + 1e-6)
        if ratio < 0.3 or ratio > 3.5:
            return float(np.clip(abs(np.log(ratio + 1e-3)) / 2.0, 0.0, 1.0))
        return 0.15
