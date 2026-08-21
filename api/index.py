import io
import base64
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

# Ensure project root in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.config import PRIMARY_CLASSES, FILTER_CLASSES, AI_SUBFAMILIES
from utils.logger import logger
from preprocessing.image_processor import ImageProcessor
from forensics import ImageForensicSuite
from models.face_detector import FaceForensicDetector
from models.ensemble import EnsembleDecisionEngine

app = FastAPI(
    title="AI Image Forensics & Filter Detection Serverless API",
    description="High-performance Serverless Computer Vision Forensics, Filter Analysis, and Authenticity Scoring.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Forensic Instances
processor = ImageProcessor()
forensic_suite = ImageForensicSuite()
face_detector = FaceForensicDetector()
ensemble_engine = EnsembleDecisionEngine()

def image_to_base64(img: Image.Image) -> str:
    """Encodes PIL Image to Base64 data URL."""
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

@app.get("/")
@app.get("/api/health")
def health_check():
    """Health check endpoint for Vercel and monitoring."""
    return {
        "status": "healthy",
        "service": "AI Image & Filter Detection System (Vercel Serverless)",
        "deployment_type": "Serverless Lambda (<500MB Optimized)",
        "supported_classes": PRIMARY_CLASSES,
        "supported_filters": FILTER_CLASSES,
        "supported_subfamilies": AI_SUBFAMILIES
    }

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Serverless image forensic analysis endpoint.
    Performs FFT frequency analysis, ELA compression forensics, noise residual maps,
    GLCM texture metrics, face smoothing forensics, and calibrated ensemble scoring.
    """
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        valid, msg, pil_img = processor.validate_image(contents)
        if not valid or pil_img is None:
            raise HTTPException(status_code=400, detail=msg)

        # 1. Metadata Extraction & Cryptographic Hashes
        metadata = processor.extract_metadata(pil_img, raw_bytes=contents)

        # 2. Computer Vision Forensics (FFT, Noise, ELA, GLCM)
        forensics_results = forensic_suite.analyze_all(pil_img)

        # 3. Face Forensics & Smoothing
        face_results = face_detector.detect_and_analyze(pil_img)

        # 4. Multi-Label Filter Heuristic Signals
        detected_filters = []
        tex = forensics_results.get("texture_analysis", {})
        comp = forensics_results.get("compression_analysis", {})
        
        if tex.get("laplacian_sharpness", 0) > 800:
            detected_filters.append({"name": "sharpening", "label": "Sharpening", "score": 0.85})
        if face_results.get("skin_smoothing_score", 0) > 0.45:
            detected_filters.append({"name": "skin_smoothing", "label": "Skin Smoothing", "score": face_results["skin_smoothing_score"]})
        if comp.get("jpeg_grid_strength", 0) > 0.5:
            detected_filters.append({"name": "compression_artifacts", "label": "Compression Artifacts", "score": 0.78})
        if tex.get("glcm_homogeneity", 0) > 0.88:
            detected_filters.append({"name": "beauty_filter", "label": "Beauty Filter", "score": 0.82})

        # 5. Multimodal Forensic Scoring
        ai_synth_score = float(forensics_results.get("composite_forensic_score", 0.0))
        simulated_ai_pred = {
            "ai_likeness_score": ai_synth_score,
            "class_probabilities": {
                "REAL": round(1.0 - ai_synth_score, 4),
                "AI_GENERATED": round(ai_synth_score, 4),
                "AI_EDITED": 0.05,
                "FILTERED": 0.10 if detected_filters else 0.0,
                "MANIPULATED": round(comp.get("ela_anomaly_score", 0.0), 4)
            },
            "top_subfamily": "Diffusion-Generated (e.g. Midjourney, SD, Flux)" if ai_synth_score > 0.5 else "Natural Photographic Capture"
        }

        ensemble_res = ensemble_engine.evaluate(
            ai_pred=simulated_ai_pred,
            filter_pred={"detected_filters": detected_filters, "max_filter_score": max([f['score'] for f in detected_filters], default=0.0)},
            forensics_pred=forensics_results,
            face_pred=face_results
        )

        # 6. Visual Artifacts Base64
        visuals = {}
        ela_arr = comp.get("ela_image")
        if ela_arr is not None:
            visuals["ela_visual"] = image_to_base64(Image.fromarray(ela_arr))
        
        noise_arr = forensics_results.get("noise_analysis", {}).get("noise_map_rgb")
        if noise_arr is not None:
            visuals["noise_map"] = image_to_base64(Image.fromarray(noise_arr))

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "verdict": ensemble_res.get("verdict"),
            "authenticity_score": ensemble_res.get("authenticity_score"),
            "confidence": ensemble_res.get("overall_confidence"),
            "is_uncertain": ensemble_res.get("is_uncertain"),
            "probabilities": ensemble_res.get("probabilities"),
            "ai_subfamily": simulated_ai_pred["top_subfamily"],
            "detected_filters": detected_filters,
            "face_analysis": {
                "detected": face_results.get("face_detected"),
                "count": face_results.get("face_count"),
                "skin_smoothing_score": face_results.get("skin_smoothing_score")
            },
            "forensics_summary": {
                "composite_score": forensics_results.get("composite_forensic_score"),
                "spectral_anomaly_score": forensics_results.get("frequency_analysis", {}).get("spectral_anomaly_score"),
                "noise_inconsistency": forensics_results.get("noise_analysis", {}).get("noise_inconsistency_score"),
                "ela_anomaly_score": comp.get("ela_anomaly_score"),
                "texture_anomaly_score": tex.get("texture_anomaly_score")
            },
            "metadata": {
                "dimensions": f"{metadata['width']}x{metadata['height']}",
                "sha256": metadata["hashes"]["sha256"],
                "phash": metadata["hashes"]["phash"]
            },
            "visuals": visuals
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Vercel API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
