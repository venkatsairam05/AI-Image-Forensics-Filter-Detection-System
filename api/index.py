import io
import base64
import sys
from pathlib import Path
from typing import Optional
from PIL import Image

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.config import PRIMARY_CLASSES, FILTER_CLASSES, AI_SUBFAMILIES
from utils.logger import logger
from pipeline import ForensicPipeline
from feedback.feedback_manager import FeedbackManager

app = FastAPI(
    title="AI Image & Filter Forensics API",
    description="REST API for Deep Learning Image Authenticity, Filter Detection, and Explainable AI.",
    version="1.0.0"
)

# Enable CORS for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
pipeline_instance = None
feedback_manager = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = ForensicPipeline()
    return pipeline_instance

def get_feedback_manager():
    global feedback_manager
    if feedback_manager is None:
        feedback_manager = FeedbackManager()
    return feedback_manager

def image_to_base64(img: Image.Image) -> str:
    """Encodes PIL Image to Base64 data URL."""
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

@app.get("/")
@app.get("/api/health")
def health_check():
    """Health check endpoint for Vercel and uptime monitors."""
    return {
        "status": "healthy",
        "service": "AI Image & Filter Detection System",
        "supported_classes": PRIMARY_CLASSES,
        "supported_filters": FILTER_CLASSES,
        "supported_subfamilies": AI_SUBFAMILIES
    }

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    generate_gradcam: bool = Form(True)
):
    """
    Analyzes an uploaded image and returns deep learning authenticity probabilities,
    filter detections, computer vision forensics, and base64 Grad-CAM overlay.
    """
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        pipeline = get_pipeline()
        result = pipeline.run_analysis(contents, generate_cam=generate_gradcam)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Image analysis failed."))

        # Convert numpy visual arrays to base64 images for web response
        visuals = {}
        if "explainability" in result and "overlay_image" in result["explainability"]:
            cam_arr = result["explainability"]["overlay_image"]
            visuals["gradcam_overlay"] = image_to_base64(Image.fromarray(cam_arr))

        if "forensics" in result:
            ela_arr = result["forensics"].get("compression_analysis", {}).get("ela_image")
            if ela_arr is not None:
                visuals["ela_visual"] = image_to_base64(Image.fromarray(ela_arr))

        # Format clean JSON output
        response_payload = {
            "success": True,
            "filename": file.filename,
            "execution_time_seconds": result.get("execution_time_seconds"),
            "verdict": result.get("ensemble", {}).get("verdict"),
            "authenticity_score": result.get("ensemble", {}).get("authenticity_score"),
            "confidence": result.get("ensemble", {}).get("overall_confidence"),
            "is_uncertain": result.get("ensemble", {}).get("is_uncertain"),
            "probabilities": result.get("ensemble", {}).get("probabilities"),
            "ai_subfamily": result.get("ai_detection", {}).get("top_subfamily"),
            "detected_filters": result.get("filter_detection", {}).get("detected_filters"),
            "face_analysis": {
                "detected": result.get("face_analysis", {}).get("face_detected"),
                "count": result.get("face_analysis", {}).get("face_count"),
                "skin_smoothing_score": result.get("face_analysis", {}).get("skin_smoothing_score")
            },
            "forensics_summary": {
                "composite_score": result.get("forensics", {}).get("composite_forensic_score"),
                "high_freq_ratio": result.get("forensics", {}).get("frequency_analysis", {}).get("high_frequency_ratio"),
                "noise_inconsistency": result.get("forensics", {}).get("noise_analysis", {}).get("noise_inconsistency_score")
            },
            "explainability": {
                "reasoning": result.get("explainability", {}).get("forensic_reasoning"),
                "disclaimer": result.get("explainability", {}).get("disclaimer")
            },
            "visuals": visuals
        }

        return JSONResponse(content=response_payload)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(
    image_hash: str = Form(...),
    predicted_verdict: str = Form(...),
    predicted_ai_prob: float = Form(...),
    predicted_confidence: float = Form(...),
    user_label: str = Form(...),
    user_agrees: bool = Form(...),
    comment: Optional[str] = Form("")
):
    """Submits user correction/verification for continuous model improvement."""
    try:
        mgr = get_feedback_manager()
        # In API mode without image re-upload, record placeholder
        entry_id = mgr.record_feedback(
            pil_img=Image.new("RGB", (64, 64), color="gray"),
            image_hash=image_hash,
            predicted_verdict=predicted_verdict,
            predicted_ai_prob=predicted_ai_prob,
            predicted_confidence=predicted_confidence,
            user_label=user_label,
            user_agrees=user_agrees,
            comment=comment or ""
        )
        return {"success": True, "feedback_id": entry_id, "message": "Feedback recorded successfully."}
    except Exception as e:
        logger.error(f"API feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
