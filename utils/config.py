import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FEEDBACK_DIR = DATA_DIR / "feedback"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_WEIGHTS_DIR = BASE_DIR / "models_weights"
DATASET_DIR = BASE_DIR / "dataset"

# Ensure runtime directories exist (fail-safe for read-only serverless)
try:
    for d in [DATA_DIR, UPLOADS_DIR, FEEDBACK_DIR, PROCESSED_DIR, MODELS_WEIGHTS_DIR, DATASET_DIR]:
        d.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
NUM_WORKERS = 0 if os.name == "nt" else 2

# Core Classification Labels
PRIMARY_CLASSES = [
    "REAL",
    "AI_GENERATED",
    "AI_EDITED",
    "FILTERED",
    "MANIPULATED"
]

PRIMARY_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(PRIMARY_CLASSES)}
IDX_TO_PRIMARY_CLASS = {idx: cls for idx, cls in enumerate(PRIMARY_CLASSES)}

# AI Subfamily / Generation Categories
AI_SUBFAMILIES = [
    "Diffusion-Generated (e.g. Midjourney, SD, Flux, DALL-E)",
    "GAN-Generated (e.g. StyleGAN, ProGAN, BigGAN)",
    "AI Face Generation (e.g. DeepFake, Swap, FaceFusion)",
    "AI Image Enhancement / Neural Upscaling",
    "Other Synthetic Generation / Procedural"
]

# Multi-label Filter Classes
FILTER_CLASSES = [
    "beauty_filter",
    "skin_smoothing",
    "color_grading",
    "hdr_enhancement",
    "sharpening",
    "blur",
    "background_replacement",
    "face_modification",
    "upscaling_enhancement",
    "compression_artifacts"
]

FILTER_CLASS_DESCRIPTIONS = {
    "beauty_filter": "AI or digital beauty touch-up filter applied",
    "skin_smoothing": "Facial skin blur and high-frequency pore smoothing",
    "color_grading": "Non-linear tone curve shift or cinematic LUT grading",
    "hdr_enhancement": "Tone-mapped dynamic range expansion",
    "sharpening": "Unsharp masking or high-frequency edge amplification",
    "blur": "Synthetic depth-of-field, gaussian or bokeh blur",
    "background_replacement": "Alpha matte composite or AI background swap",
    "face_modification": "Feature morphing (eyes, jaw, nose, expression)",
    "upscaling_enhancement": "Super-resolution or generative hallucinated detail",
    "compression_artifacts": "JPEG quantization noise and DCT blocking patterns"
}

FILTER_THRESHOLDS = {
    "beauty_filter": 0.50,
    "skin_smoothing": 0.50,
    "color_grading": 0.50,
    "hdr_enhancement": 0.50,
    "sharpening": 0.50,
    "blur": 0.50,
    "background_replacement": 0.55,
    "face_modification": 0.55,
    "upscaling_enhancement": 0.50,
    "compression_artifacts": 0.50
}

# Image Processing Specs
IMG_SIZE = 256
INPUT_SHAPE = (3, IMG_SIZE, IMG_SIZE)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Forensic Weights & Thresholds
ENSEMBLE_WEIGHTS = {
    "deep_ai_model": 0.45,
    "frequency_analysis": 0.20,
    "image_forensics": 0.20,
    "face_analysis": 0.15
}

UNCERTAINTY_THRESHOLD_LOW = 0.40
UNCERTAINTY_THRESHOLD_HIGH = 0.60

# Model Paths
AI_MODEL_PATH = MODELS_WEIGHTS_DIR / "ai_detector.pth"
FILTER_MODEL_PATH = MODELS_WEIGHTS_DIR / "filter_detector.pth"
DATABASE_PATH = FEEDBACK_DIR / "feedback_data.sqlite"
