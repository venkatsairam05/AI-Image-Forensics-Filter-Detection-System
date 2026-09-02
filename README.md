# 🛡️ AI Image Forensics & Filter Detection System

> A production-grade **Deep Learning, Computer Vision, and Explainable AI** system for detecting AI-generated images (Diffusion, GANs, DeepFakes), multi-label image filters, digital manipulations, and camera sensor anomalies.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-orange.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployment%20Ready-black.svg)](https://vercel.com)

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-image-forensics-filter-detection-system-yqtcw4osi9bnvqmr5tz.streamlit.app)

**Try it now:** <https://ai-image-forensics-filter-detection-system-yqtcw4osi9bnvqmr5tz.streamlit.app>

---

## 🌟 Key Features

1. **Multimodal AI Detection Architecture:**
   - **Primary Categories:** `REAL`, `AI_GENERATED`, `AI_EDITED`, `FILTERED`, `MANIPULATED`, `UNCERTAIN`.
   - **AI Generator Subfamily Estimation:** Diffusion models (Midjourney, Stable Diffusion, DALL-E, Flux), GANs (StyleGAN, ProGAN), Face Generation (FaceSwap, DeepFake), and Neural Enhancement / Super-Resolution.
2. **Multi-Label Filter & Transformation Classifier:**
   - Simultaneously detects 10 manipulations: Beauty Filter, Skin Smoothing, Color Grading, HDR Enhancement, Sharpening, Blur, Background Replacement, Face Modification, Upscaling, and Compression Artifacts.
3. **Computer Vision Forensic Engine:**
   - **2D Fast Fourier Transform (FFT) & DCT:** Detects periodic upsampling grids and high-frequency spectral roll-offs.
   - **Error Level Analysis (ELA):** Identifies regional compression delta mismatches (spliced/inpainted regions).
   - **Sensor Noise Residual Inconsistency:** Evaluates local block-wise Poisson photon noise consistency.
   - **Texture & Edge Analysis:** Quantifies GLCM spatial co-occurrence, Local Binary Patterns (LBP) entropy, and Laplacian gradient sharpness.
4. **Explainable AI (Grad-CAM & Grad-CAM++):**
   - Renders high-resolution activation heatmaps, alpha-blended overlays, and automated forensic reasoning text explaining *why* the model made its decision.
5. **Court/Forensic-Grade PDF Reports:**
   - One-click downloadable intelligence PDF report generated with ReportLab containing embedded visual plates, probability distributions, cryptographic hashes, and authenticity seals.
6. **Continuous Learning & Retraining Pipeline:**
   - SQLite-backed Human-in-the-Loop feedback queue with validation safeguards and automated champion-candidate model evaluation before promotion.
7. **Deployment Ready:**
   - Interactive multi-page **Streamlit Web Application**.
   - Serverless REST API with **FastAPI** configured for **Vercel** (`vercel.json`), Docker containerization (`Dockerfile` & `docker-compose.yml`), and Streamlit Community Cloud.

---

## 🏗️ Architecture & Pipeline Flow

```text
User Image Upload (JPG, PNG, WEBP)
        │
        ▼
[1. Preprocessing & Hashing] ───► EXIF Metadata, SHA-256, pHash, dHash
        │
        ├──► [2. CV Forensics Engine] ───► FFT/DCT + ELA + Noise Residual + GLCM/LBP
        ├──► [3. Facial Forensic ROI] ───► Skin Smoothing Index + Boundary Discrepancy
        ├──► [4. Deep Vision Model]   ───► Transfer Learning Backbone (EfficientNet)
        │         │
        │         ├──► Primary Authenticity Classifier (5 Classes)
        │         └──► Generator Subfamily Estimator (5 Categories)
        │
        ├──► [5. Filter Network]     ───► Multi-Label Sigmoid Classifier (10 Filters)
        └──► [6. Grad-CAM Engine]    ───► Heatmap + Attention Overlay + Reasoning
                  │
                  ▼
      [7. Multimodal Ensemble Layer] ───► Dynamic Weighting & Uncertainty Calibration
                  │
                  ▼
      [8. Authenticity Intelligence Report]
            ├── Streamlit Interactive UI
            ├── PDF Forensic Intelligence Report
            └── Vercel / FastAPI REST Endpoint
                  │
                  ▼
      [9. Continuous Improvement Loop] ───► Feedback Queue ──► Guarded Retraining
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
cd ai_image_forensics
pip install -r requirements.txt
```

### 2. Initialize Baseline Model Weights

Bootstrap calibrated model weights and procedural datasets in one command:

```bash
python weights_initializer.py
```

### 3. Launch Streamlit Web UI

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Vercel & Cloud

### Deploy to Vercel

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Deploy the serverless FastAPI backend:
   ```bash
   vercel
   ```
   Vercel automatically detects `vercel.json` and configures `api/index.py` as Python serverless functions.

### Deploy with Docker

```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:8501`.

### Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Link your repository at [share.streamlit.io](https://share.streamlit.io).
3. Set the main file path to `app.py`.

---

## 📡 REST API Reference

### Health Check
```http
GET /api/health
```

### Analyze Image
```http
POST /api/analyze
Content-Type: multipart/form-data

file: <image_file>
generate_gradcam: true
```

**Response Example:**
```json
{
  "success": true,
  "filename": "sample.jpg",
  "verdict": "AI-GENERATED",
  "authenticity_score": 14.2,
  "confidence": 85.8,
  "is_uncertain": false,
  "probabilities": {
    "real": 14.2,
    "ai_generated": 85.8,
    "ai_edited": 0.0,
    "filtered": 0.0,
    "manipulated": 0.0
  },
  "ai_subfamily": "Diffusion-Generated (e.g. Midjourney, SD, Flux, DALL-E)",
  "detected_filters": [
    {
      "name": "skin_smoothing",
      "label": "Skin Smoothing",
      "score": 0.82
    }
  ],
  "visuals": {
    "gradcam_overlay": "data:image/jpeg;base64,...",
    "ela_visual": "data:image/jpeg;base64,..."
  }
}
```

### Submit Feedback
```http
POST /api/feedback
Content-Type: application/x-www-form-urlencoded

image_hash=...&predicted_verdict=AI-GENERATED&predicted_ai_prob=85.8&predicted_confidence=85.8&user_label=AI_GENERATED&user_agrees=true&comment=Midjourney_v6
```

---

## 📊 Model Training & Evaluation

### Train AI Classifier
```bash
python training/train_ai_detector.py
```

### Train Multi-Label Filter Detector
```bash
python training/train_filter_detector.py
```

### Evaluate Model Performance
```bash
python training/evaluate.py
```

### Recommended Public Datasets
- **CIFAKE:** Real vs Synthetic benchmark dataset.
- **GenImage:** Multimodal benchmark across Midjourney, SD, DALL-E, and VQ-GAN.
- **FaceForensics++:** DeepFake, FaceSwap, and NeuralTextures forensics.
- **Synthbuster:** Cross-generator synthetic evaluation dataset.

---

## 🧪 Running Automated Tests

Run the complete test suite:

```bash
python -m unittest tests/test_pipeline.py
```

---

## 🔒 Security & Privacy

- **Local Execution:** Uploaded files are processed in-memory and deleted immediately unless the user explicitly opts into the feedback collection pool.
- **Data Protection:** No data is sent to third-party APIs.
- **Forensic Transparency:** Model confidence and Grad-CAM activations are presented as model-based indicators rather than absolute legal proofs.
