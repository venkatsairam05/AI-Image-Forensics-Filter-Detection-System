import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# System & Local Modules
from utils.config import (
    PRIMARY_CLASSES, FILTER_CLASSES, FILTER_CLASS_DESCRIPTIONS,
    AI_SUBFAMILIES, DEVICE, DATASET_DIR, MODELS_WEIGHTS_DIR
)
from utils.logger import logger
from pipeline import ForensicPipeline
from feedback.feedback_manager import FeedbackManager
from feedback.dataset_manager import DatasetManager
from feedback.retrain_pipeline import RetrainingPipeline
from training.evaluate import ModelEvaluator

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI Image Forensics & Filter Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CINEMATIC FORENSIC UI — animated glassmorphism design system
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Animated aurora background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(109, 92, 255, 0.22), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(0, 212, 255, 0.16), transparent 55%),
        radial-gradient(900px 700px at 60% 110%, rgba(255, 78, 205, 0.14), transparent 60%),
        linear-gradient(160deg, #0b0f1a 0%, #0c1122 45%, #0a0e1c 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: rgba(11, 15, 26, 0.55); backdrop-filter: blur(8px); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(16, 22, 43, 0.9), rgba(11, 15, 26, 0.95));
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"]::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
    background: linear-gradient(90deg, #6d5cff, #00d4ff, #ff4ecd, #6d5cff);
    background-size: 300% 100%;
    animation: gradient-x 6s linear infinite;
}

/* Floating animated orbs */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.55;
    pointer-events: none;
    z-index: 0;
}
.orb-1 { width: 420px; height: 420px; left: -120px; top: 8%;  background: #6d5cff; animation: float-a 16s ease-in-out infinite; }
.orb-2 { width: 360px; height: 360px; right: -100px; top: 40%; background: #00d4ff; animation: float-b 20s ease-in-out infinite; }
.orb-3 { width: 300px; height: 300px; left: 38%; bottom: -140px; background: #ff4ecd; animation: float-c 24s ease-in-out infinite; }

@keyframes float-a { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(60px,40px) scale(1.15)} }
@keyframes float-b { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-70px,-50px) scale(1.2)} }
@keyframes float-c { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(40px,-60px) scale(1.1)} }

@keyframes gradient-x { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* Headers — animated gradient text */
.main-header {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #a78bfa, #22d3ee, #f0abfc, #a78bfa);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: gradient-x 8s linear infinite;
    filter: drop-shadow(0 4px 22px rgba(139, 92, 246, 0.35));
}
.sub-header {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    animation: fadeUp .6s ease both;
}

/* Verdict banner — animated gradient + shine sweep */
.verdict-banner {
    position: relative;
    overflow: hidden;
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 12px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.06);
    animation: banner-in .7s cubic-bezier(.2,.9,.3,1.2) both;
}
.verdict-banner::before {
    content: ""; position: absolute; top: 0; left: -120%;
    width: 60%; height: 100%;
    background: linear-gradient(105deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: skewX(-20deg);
    animation: shine 3.2s ease-in-out infinite;
}
.verdict-banner::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    background-size: 300% 100%;
    animation: gradient-x 6s linear infinite;
}
@keyframes banner-in { from{opacity:0; transform:translateY(-14px) scale(.97)} to{opacity:1; transform:translateY(0) scale(1)} }
@keyframes shine { 0%,55%{left:-120%} 100%{left:130%} }

/* Metric cards — pulsing glass, hover lift */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeUp .6s ease both;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.02);
    border-color: rgba(167,139,250,0.6);
    box-shadow: 0 16px 40px rgba(109,92,255,0.28);
}
[data-testid="stMetricValue"] {
    background: linear-gradient(90deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 800;
}
[data-testid="stMetricLabel"] { color: #94a3b8; font-weight: 600; }

/* Filter badges — gradient pulse */
.filter-badge {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    background: linear-gradient(120deg, #6d5cff, #8b5cf6, #00d4ff);
    background-size: 220% 100%;
    animation: gradient-x 5s linear infinite, badge-in .5s ease both;
    box-shadow: 0 4px 16px rgba(109,92,255,0.4);
    border: 1px solid rgba(255,255,255,0.2);
}
@keyframes badge-in { from{opacity:0; transform:scale(.6)} to{opacity:1; transform:scale(1)} }

/* Buttons — gradient glow */
.stButton > button, [data-testid="stDownloadButton"] > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(167,139,250,0.5);
    background: linear-gradient(120deg, #6d5cff, #8b5cf6 50%, #00d4ff);
    background-size: 220% 100%;
    color: white;
    font-weight: 700;
    box-shadow: 0 8px 24px rgba(109,92,255,0.35);
    transition: all .25s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    background-position: 100% 0;
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 32px rgba(109,92,255,0.5);
}

/* Inputs / file uploader glass */
[data-testid="stFileUploader"] { border-radius: 14px; overflow: hidden; }
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(167,139,250,0.5);
    border-radius: 14px;
    transition: all .25s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: #22d3ee; background: rgba(109,92,255,0.08); }

/* Selectbox / inputs */
.stSelectbox [data-baseweb="select"] > div, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    border-color: rgba(255,255,255,0.12) !important;
}
.stSelectbox [data-baseweb="popover"] ul { background: #10162b !important; }

/* Radio pills */
.stRadio [role="radiogroup"] > label {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 9999px;
    padding: 0.25rem 0.9rem;
    transition: all .2s ease;
}
.stRadio [role="radiogroup"] > label:hover { border-color: #6d5cff; }

/* Tabs — animated underline */
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    transition: color .2s ease, transform .2s ease;
}
[data-testid="stTabs"] button:hover { color: #e7ecf7 !important; transform: translateY(-1px); }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    font-weight: 700;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #6d5cff, #00d4ff) !important;
    animation: gradient-x 4s linear infinite;
}

/* Expanders */
[data-testid="stExpander"] details {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    transition: border-color .25s ease;
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* Images — forensic scanline sweep */
[data-testid="stImage"] { position: relative; border-radius: 14px; overflow: hidden; }
[data-testid="stImage"]::after {
    content: "";
    position: absolute; left: 0; right: 0; top: -15%;
    height: 34%;
    background: linear-gradient(180deg, transparent, rgba(0,212,255,0.22), transparent);
    animation: scan 3.5s linear infinite;
    pointer-events: none;
    z-index: 2;
}
@keyframes scan { 0%{top:-15%} 100%{top:115%} }
[data-testid="stImage"] img { border-radius: 14px; }

/* Chart frames */
[data-testid="stPlotlyChart"] { border-radius: 14px; overflow: hidden; }
[data-testid="stPlotlyChart"]:hover { box-shadow: 0 10px 30px rgba(109,92,255,0.18); }

/* Progress bar */
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #6d5cff, #00d4ff, #ff4ecd);
    background-size: 300% 100%;
    animation: gradient-x 4s linear infinite;
}

/* Dataframes / json */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stJson"] { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; }

/* Info / success / warning — glass toast */
.stAlert { border-radius: 12px !important; backdrop-filter: blur(8px); }

/* Disclaimer box */
.disclaimer-box {
    background: rgba(245, 158, 11, 0.12);
    border-left: 4px solid #f59e0b;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #fcd34d;
    margin-top: 1rem;
    animation: fadeUp .6s ease .15s both;
}

/* Metric-card fallback (kept for legacy sections) */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform .25s ease, box-shadow .25s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(109,92,255,0.25); }

/* Fade-up entrance */
@keyframes fadeUp { from{opacity:0; transform:translateY(16px)} to{opacity:1; transform:translateY(0)} }

/* Custom scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6d5cff, #00d4ff);
    border-radius: 9999px;
}
</style>
""", unsafe_allow_html=True)

# Floating animation orbs (background layer)
st.markdown("""
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# Cinematic JS — animated metric counters + stagger reveals
st.markdown("""
<script>
(function () {
    try {
        function animateValue(el, target) {
            var dur = 900, start = null;
            var numeric = parseFloat(String(target).replace(/[^0-9.-]/g, "")) || 0;
            var suffix = String(target).replace(/[0-9.,%]+/g, "").trim();
            function tick(ts) {
                if (!start) start = ts;
                var p = Math.min((ts - start) / dur, 1);
                var eased = 1 - Math.pow(1 - p, 3);
                var val = numeric * eased;
                var txt = (Number.isInteger(numeric) ? Math.round(val).toString() : val.toFixed(1)) + (suffix ? " " + suffix : "");
                el.textContent = txt;
                if (p < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
        }

        var seen = new WeakSet();
        function wireCounters() {
            document.querySelectorAll('[data-testid="stMetricValue"]').forEach(function (el) {
                if (seen.has(el)) return;
                seen.add(el);
                var target = el.textContent;
                if (!target || /^[A-Za-z]/.test(target)) return;
                try {
                    animateValue(el, target);
                } catch (e) {}
            });
        }

        function wireReveals() {
            var col = document.querySelector('[data-testid="stMain"]')
                        || document.querySelector('.block-container');
            if (!col) return;
            col.querySelectorAll(':scope > div').forEach(function (el, i) {
                if (getComputedStyle(el).animationName === 'none' || el.getAttribute('data-rv')) return;
                el.setAttribute('data-rv', '1');
                el.style.animation = 'fadeUp .6s ease ' + (i * 0.06) + 's both';
            });
        }

        var t = null;
        new MutationObserver(function () {
            clearTimeout(t);
            t = setTimeout(function () {
                try { wireCounters(); wireReveals(); } catch (e) {}
            }, 120);
        }).observe(document.body, { childList: true, subtree: true });

        setTimeout(function () { try { wireCounters(); wireReveals(); } catch (e) {} }, 400);
    } catch (e) {}
})();
</script>
""", unsafe_allow_html=True)

# Cache Pipeline Instance in Streamlit Session
@st.cache_resource
def load_pipeline():
    return ForensicPipeline()

@st.cache_resource
def load_feedback_mgr():
    return FeedbackManager()

@st.cache_resource
def load_dataset_mgr():
    return DatasetManager()

pipeline = load_pipeline()
feedback_mgr = load_feedback_mgr()
dataset_mgr = load_dataset_mgr()

# Initialize session states
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "current_img" not in st.session_state:
    st.session_state.current_img = None

# Sidebar Navigation
st.sidebar.markdown("## 🛡️ **AI Image Forensics**")
st.sidebar.caption("Deep Learning • Forensics • Grad-CAM")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Analyze Image",
        "💡 Explainability (Grad-CAM)",
        "🔬 Forensics Deep-Dive",
        "📂 Batch Analysis",
        "📊 Model Performance & Admin",
        "ℹ️ System Architecture & Datasets"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.text(f"Device: {DEVICE.upper()}")
st.sidebar.text("Model: EfficientNet-B0")
st.sidebar.text(f"Images Analyzed: {len(st.session_state.analysis_history)}")
feedback_stats = feedback_mgr.get_feedback_stats()
st.sidebar.text(f"User Feedback: {feedback_stats.get('total_submissions', 0)} entries")

# ==========================================
# PAGE 1: ANALYZE IMAGE
# ==========================================
if menu == "🔍 Analyze Image":
    st.markdown('<div class="main-header">AI Image & Filter Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload any image to verify authenticity, identify AI generation subfamilies, and detect digital filters.</div>', unsafe_allow_html=True)

    # Top Control Bar
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Images are processed locally and securely."
        )

    with col_sample:
        st.markdown("**Or load a sample:**")
        sample_choice = st.selectbox(
            "Select Sample",
            ["None", "Procedural Real Photo", "Procedural AI-Generated", "Filtered Portrait"]
        )

    # Handle image source
    pil_to_process = None
    file_bytes = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        pil_to_process = Image.open(io.BytesIO(file_bytes))
    elif sample_choice != "None":
        from training.dataset_generator import SyntheticDatasetGenerator
        from preprocessing.augmentation import apply_forensic_augmentations
        base = SyntheticDatasetGenerator().create_base_sample(seed=42)
        if sample_choice == "Procedural Real Photo":
            pil_to_process = apply_forensic_augmentations(base, mode="gaussian_noise")
        elif sample_choice == "Procedural AI-Generated":
            pil_to_process = apply_forensic_augmentations(base, mode="skin_smoothing")
            pil_to_process = apply_forensic_augmentations(pil_to_process, mode="sharpening")
        elif sample_choice == "Filtered Portrait":
            pil_to_process = apply_forensic_augmentations(base, mode="color_grading")

    if pil_to_process is not None:
        # Run Pipeline
        with st.spinner("Analyzing image via Deep Neural Network, FFT spectrum, ELA compression, and Grad-CAM..."):
            res = pipeline.run_analysis(pil_to_process, generate_cam=True)

        if res.get("success"):
            st.session_state.current_analysis = res
            st.session_state.current_img = pil_to_process
            
            # Record in history if unique
            h = res["metadata"]["hashes"]["sha256"]
            if not any(entry["hash"] == h for entry in st.session_state.analysis_history):
                st.session_state.analysis_history.insert(0, {
                    "hash": h,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "verdict": res["ensemble"]["verdict"],
                    "authenticity": res["ensemble"]["authenticity_score"],
                    "confidence": res["ensemble"]["overall_confidence"]
                })

            ensemble = res["ensemble"]
            ai_det = res["ai_detection"]
            filt_det = res["filter_detection"]
            face_det = res["face_analysis"]
            meta = res["metadata"]

            # Main Result Section
            st.markdown("---")
            
            # Verdict Banner
            banner_bg = ensemble.get("verdict_color", "#10b981")
            st.markdown(f"""
            <div class="verdict-banner" style="background-color: {banner_bg};">
                <h2 style="margin:0; font-size:1.8rem;">{ensemble.get('verdict_badge', 'Result')}</h2>
                <p style="margin:0.4rem 0 0 0; font-size:1.05rem; opacity:0.95;">{ensemble.get('summary')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Two Column Display
            col_img, col_metrics = st.columns([1.1, 1.3])

            with col_img:
                st.markdown("### 🖼️ Inspected Image")
                # Show annotated face box if detected
                if face_det.get("face_detected"):
                    st.image(face_det["annotated_image"], caption="Detected Face ROI & Boundary", use_container_width=True)
                else:
                    st.image(pil_to_process, caption="Original Input Image", use_container_width=True)

                # Metadata Expandable
                with st.expander("📋 Image Metadata & Hashes", expanded=False):
                    st.write(f"**Dimensions:** {meta['width']} x {meta['height']} px")
                    st.write(f"**Format:** {meta['format']} ({meta['mode']})")
                    st.write(f"**Camera Make/Model:** {meta['camera_make']} / {meta['camera_model']}")
                    st.write(f"**Software:** {meta['software']}")
                    if meta.get("ai_software_indicator"):
                        st.error("⚠️ AI Generator software signature detected in metadata!")
                    st.code(f"SHA-256: {meta['hashes']['sha256']}\npHash:   {meta['hashes']['phash']}")

            with col_metrics:
                st.markdown("### 📊 Forensic Intelligence Scorecard")
                
                # 4 KPI metric cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Authenticity Score", f"{ensemble['authenticity_score']}%", help="Higher score indicates real, untouched optical capture.")
                m2.metric("Overall Confidence", f"{ensemble['overall_confidence']}%", help="Model prediction confidence.")
                m3.metric("AI-Likeness Index", f"{round(ai_det['ai_likeness_score']*100, 1)}%")

                st.markdown("#### 🎯 Class Probability Distribution")
                probs = ensemble["probabilities"]
                prob_df = pd.DataFrame({
                    "Class": ["Real", "AI-Generated", "AI-Edited", "Filtered", "Manipulated"],
                    "Probability (%)": [
                        probs.get("real", 0),
                        probs.get("ai_generated", 0),
                        probs.get("ai_edited", 0),
                        probs.get("filtered", 0),
                        probs.get("manipulated", 0)
                    ]
                })

                fig_probs = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Class",
                    orientation="h",
                    color="Class",
                    color_discrete_map={
                        "Real": "#10b981",
                        "AI-Generated": "#ef4444",
                        "AI-Edited": "#ec4899",
                        "Filtered": "#3b82f6",
                        "Manipulated": "#8b5cf6"
                    },
                    text="Probability (%)"
                )
                fig_probs.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_probs, use_container_width=True)

                # AI Generation Category
                st.markdown("#### 🤖 Estimated AI Architecture / Subfamily")
                st.info(f"**Likely Generator Type:** {ai_det.get('top_subfamily', 'N/A')}")

                # Detected Filters
                st.markdown("#### ✨ Multi-Label Filter Detections")
                detected_filters = filt_det.get("detected_filters", [])
                if detected_filters:
                    pills_html = "".join([
                        f'<span class="filter-badge">🏷️ {f["label"]} ({int(f["score"]*100)}%)</span>'
                        for f in detected_filters
                    ])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.success("No heavy digital filters or beauty enhancements detected.")

            # PDF Download & Explainability Summary
            st.markdown("---")
            col_exp_sum, col_pdf = st.columns([2, 1])

            with col_exp_sum:
                st.markdown("### 💡 Explainable AI Summary")
                st.write(res.get("explainability", {}).get("forensic_reasoning", "Analysis complete."))
                st.caption(f"Execution time: {res['execution_time_seconds']} seconds on {DEVICE.upper()}.")

            with col_pdf:
                st.markdown("### 📄 Forensic Report")
                try:
                    pdf_bytes = pipeline.generate_pdf_report(res)
                    st.download_button(
                        label="📥 Download Forensic PDF Report",
                        data=pdf_bytes,
                        file_name=f"forensic_report_{res['metadata']['hashes']['sha256'][:8]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation note: {e}")

            # Feedback Submission Section
            st.markdown("---")
            st.markdown("### 🗳️ Continuous Learning: Human-in-the-Loop Feedback")
            st.caption("Provide feedback to help improve the model. Submissions are staged for admin verification before retraining.")

            with st.form("user_feedback_form"):
                fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 2])
                with fb_c1:
                    is_correct = st.radio("Is this prediction correct?", ["Yes", "No"], horizontal=True)
                with fb_c2:
                    true_category = st.selectbox("True Image Category", PRIMARY_CLASSES, index=0)
                with fb_c3:
                    feedback_comment = st.text_input("Comments / Context (optional)", placeholder="e.g. Midjourney v6 with prompt XYZ")

                submit_fb = st.form_submit_button("Submit Feedback to Continuous Learning Pool")

                if submit_fb:
                    entry_id = feedback_mgr.record_feedback(
                        pil_img=pil_to_process,
                        image_hash=meta["hashes"]["sha256"],
                        predicted_verdict=ensemble["verdict"],
                        predicted_ai_prob=ensemble["probabilities"]["ai_generated"],
                        predicted_confidence=ensemble["overall_confidence"],
                        user_label=true_category,
                        user_agrees=(is_correct == "Yes"),
                        comment=feedback_comment,
                        metadata={"dimensions": f"{meta['width']}x{meta['height']}"}
                    )
                    st.success(f"✅ Feedback #{entry_id} successfully recorded! Thank you for contributing to model improvement.")

        else:
            st.error(f"Image analysis error: {res.get('error')}")

    else:
        st.info("👈 Please upload an image or choose a sample above to begin analysis.")

# ==========================================
# PAGE 2: EXPLAINABILITY (GRAD-CAM)
# ==========================================
elif menu == "💡 Explainability (Grad-CAM)":
    st.markdown('<div class="main-header">Explainable AI: Grad-CAM Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Gradient-weighted Class Activation Mapping visualizes which visual features influenced the neural network.</div>', unsafe_allow_html=True)

    if st.session_state.current_analysis is not None and st.session_state.current_img is not None:
        res = st.session_state.current_analysis
        orig_img = st.session_state.current_img
        explain = res.get("explainability", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. Input Image")
            st.image(orig_img, use_container_width=True)

        with c2:
            st.markdown("#### 2. Grad-CAM Heatmap")
            if "heatmap_image" in explain:
                st.image(explain["heatmap_image"], use_container_width=True)

        with c3:
            st.markdown("#### 3. Attention Overlay")
            if "overlay_image" in explain:
                st.image(explain["overlay_image"], use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔬 Forensic Activation Interpretation")
        st.info(f"**Model Reasoning:** {explain.get('forensic_reasoning', 'N/A')}")
        
        st.markdown("""
        <div class="disclaimer-box">
            <b>Explainability Transparency:</b> High-temperature regions (Red/Yellow) denote visual spatial locations that generated maximal gradient backpropagation for the predicted class. Blue regions had negligible influence.
        </div>
        """, unsafe_allow_html=True)

        # Active Hotspot coordinates
        hotspots = explain.get("hotspots", [])
        if hotspots:
            st.markdown("#### 📍 Peak Activation Regions (Hotspots)")
            st.dataframe(pd.DataFrame(hotspots), use_container_width=True)

    else:
        st.warning("⚠️ No image analyzed yet. Please run an analysis in the 'Analyze Image' tab first.")

# ==========================================
# PAGE 3: FORENSICS DEEP-DIVE
# ==========================================
elif menu == "🔬 Forensics Deep-Dive":
    st.markdown('<div class="main-header">Computer Vision Image Forensics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into 2D Fast Fourier Transforms (FFT), Error Level Analysis (ELA), Noise residuals, and Texture GLCM.</div>', unsafe_allow_html=True)

    if st.session_state.current_analysis is not None and st.session_state.current_img is not None:
        res = st.session_state.current_analysis
        forensics = res.get("forensics", {})
        meta = res.get("metadata", {})

        tab_fft, tab_ela, tab_noise, tab_texture, tab_exif = st.tabs([
            "🌊 Frequency Spectrum (FFT)",
            "🧱 Error Level Analysis (ELA)",
            "📡 Sensor Noise Variance",
            "🧶 Texture & Edge Forensics",
            "📜 EXIF & Hash Forensics"
        ])

        # 1. FFT
        with tab_fft:
            st.markdown("### 2D Fast Fourier Transform & Azimuthal Radial Profile")
            st.caption("AI generators frequently leave high-frequency grid artifacts or abnormal spectral power roll-offs.")
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                fft_img = forensics.get("frequency_analysis", {}).get("spectrum_image")
                if fft_img is not None:
                    st.image(fft_img, caption="Centered 2D FFT Magnitude Spectrum", use_container_width=True)
            
            with f_col2:
                radial = forensics.get("frequency_analysis", {}).get("radial_profile", [])
                if radial:
                    rad_df = pd.DataFrame({"Frequency Radius": list(range(len(radial))), "Spectral Power": radial})
                    fig_rad = px.line(rad_df, x="Frequency Radius", y="Spectral Power", title="Azimuthal Radial Power Distribution")
                    st.plotly_chart(fig_rad, use_container_width=True)

                st.metric("High-Frequency Energy Ratio", f"{forensics.get('frequency_analysis', {}).get('high_frequency_ratio', 0.0)}")
                st.metric("Spectral Anomaly Index", f"{forensics.get('frequency_analysis', {}).get('spectral_anomaly_score', 0.0)}")

        # 2. ELA
        with tab_ela:
            st.markdown("### Error Level Analysis (ELA)")
            st.caption("Highlights compression rate discrepancies between pristine and spliced/inpainted image regions.")
            
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                ela_img = forensics.get("compression_analysis", {}).get("ela_image")
                if ela_img is not None:
                    st.image(ela_img, caption="Amplified Error Level Map", use_container_width=True)
            
            with e_col2:
                comp = forensics.get("compression_analysis", {})
                st.metric("Mean ELA Error", f"{comp.get('mean_ela_error', 0.0)}")
                st.metric("ELA Standard Deviation", f"{comp.get('std_ela_error', 0.0)}")
                st.metric("JPEG 8x8 Grid Strength", f"{comp.get('jpeg_grid_strength', 0.0)}")
                st.metric("Compression Anomaly Score", f"{comp.get('ela_anomaly_score', 0.0)}")

        # 3. Noise
        with tab_noise:
            st.markdown("### Sensor Residual Noise & Spatial Inconsistency")
            st.caption("Authentic camera sensors produce uniform Poisson noise. Generative models produce inconsistent or over-smoothed noise residuals.")
            
            n_col1, n_col2 = st.columns(2)
            with n_col1:
                noise_map = forensics.get("noise_analysis", {}).get("noise_map_rgb")
                if noise_map is not None:
                    st.image(noise_map, caption="Local Noise Inconsistency Heatmap", use_container_width=True)
            
            with n_col2:
                noise_dat = forensics.get("noise_analysis", {})
                st.metric("Global Noise Standard Dev", f"{noise_dat.get('global_noise_std', 0.0)}")
                st.metric("Normalized Inconsistency", f"{noise_dat.get('normalized_inconsistency', 0.0)}")
                st.metric("Laplacian Noise Variance", f"{noise_dat.get('laplacian_noise_var', 0.0)}")
                st.metric("Noise Anomaly Score", f"{noise_dat.get('noise_inconsistency_score', 0.0)}")

        # 4. Texture
        with tab_texture:
            st.markdown("### Gray-Level Co-occurrence (GLCM) & Local Binary Patterns (LBP)")
            tex = forensics.get("texture_analysis", {})
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                radar_df = pd.DataFrame(dict(
                    r=[
                        tex.get("glcm_homogeneity", 0.0) * 100,
                        tex.get("glcm_contrast", 0.0) * 20,
                        tex.get("glcm_energy", 0.0) * 100,
                        tex.get("lbp_entropy", 0.0) * 20,
                        min(100, tex.get("laplacian_sharpness", 0.0) / 10)
                    ],
                    theta=['Homogeneity', 'Contrast', 'Energy', 'LBP Entropy', 'Edge Sharpness']
                ))
                fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True)
                fig_radar.update_traces(fill='toself')
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with t_col2:
                st.metric("Laplacian Edge Sharpness", f"{tex.get('laplacian_sharpness', 0.0)}")
                st.metric("LBP Texture Entropy", f"{tex.get('lbp_entropy', 0.0)}")
                st.metric("GLCM Homogeneity", f"{tex.get('glcm_homogeneity', 0.0)}")
                st.metric("Texture Anomaly Score", f"{tex.get('texture_anomaly_score', 0.0)}")

        # 5. EXIF
        with tab_exif:
            st.markdown("### EXIF Metadata & Perceptual Hashing")
            st.json(meta)

    else:
        st.warning("⚠️ No image analyzed yet. Please run an analysis in the 'Analyze Image' tab first.")

# ==========================================
# PAGE 4: BATCH ANALYSIS
# ==========================================
elif menu == "📂 Batch Analysis":
    st.markdown('<div class="main-header">Batch Image Forensics Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload multiple images simultaneously for high-throughput batch auditing and CSV reporting.</div>', unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if batch_files:
        if st.button("🚀 Process All Images in Batch"):
            results_list = []
            prog_bar = st.progress(0)
            status_text = st.empty()

            for i, f in enumerate(batch_files):
                status_text.text(f"Processing {i+1}/{len(batch_files)}: {f.name}")
                raw_bytes = f.read()
                res = pipeline.run_analysis(raw_bytes, generate_cam=False)
                if res.get("success"):
                    ens = res["ensemble"]
                    ai = res["ai_detection"]
                    flt = res["filter_detection"]
                    results_list.append({
                        "Filename": f.name,
                        "Verdict": ens.get("verdict"),
                        "Authenticity Score (%)": ens.get("authenticity_score"),
                        "AI Probability (%)": ens.get("probabilities", {}).get("ai_generated"),
                        "Real Probability (%)": ens.get("probabilities", {}).get("real"),
                        "Confidence (%)": ens.get("overall_confidence"),
                        "AI Subfamily": ai.get("top_subfamily"),
                        "Filters Detected": ", ".join(flt.get("detected_filter_names", [])) or "None",
                        "Time (s)": res.get("execution_time_seconds")
                    })
                prog_bar.progress((i + 1) / len(batch_files))

            status_text.success(f"✅ Successfully processed {len(results_list)} images!")
            batch_df = pd.DataFrame(results_list)
            st.dataframe(batch_df, use_container_width=True)

            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Analysis CSV",
                data=csv_data,
                file_name="batch_forensics_report.csv",
                mime="text/csv"
            )

# ==========================================
# PAGE 5: MODEL PERFORMANCE & ADMIN
# ==========================================
elif menu == "📊 Model Performance & Admin":
    st.markdown('<div class="main-header">Model Performance & Continuous Learning Admin</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect model evaluation metrics, ROC-AUC curves, dataset statistics, and trigger guarded retraining.</div>', unsafe_allow_html=True)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_dataset(split="test")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    m_col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    m_col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    m_col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
    m_col5.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")

    st.markdown("---")
    adm_col1, adm_col2 = st.columns(2)

    with adm_col1:
        st.markdown("### 🧩 Confusion Matrix")
        cm_data = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm_data,
            x=PRIMARY_CLASSES,
            y=PRIMARY_CLASSES,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with adm_col2:
        st.markdown("### 📦 Dataset Distribution")
        ds_stats = dataset_mgr.get_dataset_stats()
        train_counts = ds_stats.get("splits", {}).get("train", {}).get("counts", {})
        ds_df = pd.DataFrame({
            "Class": list(train_counts.keys()),
            "Sample Count": list(train_counts.values())
        })
        fig_ds = px.bar(ds_df, x="Class", y="Sample Count", color="Class")
        fig_ds.update_layout(showlegend=False)
        st.plotly_chart(fig_ds, use_container_width=True)

    # Retraining & Feedback Review
    st.markdown("---")
    st.markdown("### 🔄 Continuous Retraining Pipeline")
    st.caption("Promotes candidate model only if validation accuracy strictly beats current champion.")

    col_retrain, col_fb_queue = st.columns([1, 1])

    with col_retrain:
        if st.button("🚀 Trigger Model Retraining & Benchmark"):
            with st.spinner("Synchronizing verified feedback, fine-tuning candidate model, and evaluating against champion..."):
                retrainer = RetrainingPipeline()
                retrain_res = retrainer.run_retraining(epochs=3)
                if retrain_res["is_promoted"]:
                    st.success(f"🎉 {retrain_res['status_message']}")
                else:
                    st.warning(f"ℹ️ {retrain_res['status_message']}")
                st.json(retrain_res)

    with col_fb_queue:
        st.markdown("#### 📥 Staged Feedback Queue")
        feedback_items = feedback_mgr.get_all_feedback(limit=10)
        if feedback_items:
            st.dataframe(pd.DataFrame(feedback_items)[["id", "predicted_verdict", "user_label", "user_agrees", "verified_by_admin", "status"]], use_container_width=True)
        else:
            st.info("No feedback entries submitted yet.")

# ==========================================
# PAGE 6: ARCHITECTURE & DATASETS
# ==========================================
elif menu == "ℹ️ System Architecture & Datasets":
    st.markdown('<div class="main-header">System Architecture & Public Datasets</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive guide to the multi-stage detection pipeline, forensic algorithms, and public training datasets.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ Multimodal Pipeline Architecture
    1. **Preprocessing & Verification:** Input integrity validation, aspect-ratio resizing, ImageNet normalization, EXIF extraction, and pHash computation.
    2. **Deep Learning Vision Backbone:** Dual-head EfficientNet-B0 transfer learning network classifying 5 primary categories (`REAL`, `AI_GENERATED`, `AI_EDITED`, `FILTERED`, `MANIPULATED`) and 5 generation subfamilies (`Diffusion`, `GAN`, `FaceSwap`, `Enhancement`, `Other`).
    3. **Multi-Label Filter Network:** Detects 10 simultaneous manipulations (Beauty filter, Skin smoothing, Color grading, HDR, Sharpening, Blur, Background replacement, Face modification, Upscaling, Compression).
    4. **Computer Vision Forensics:**
       - **2D Fast Fourier Transform (FFT):** Identifies checkerboard grid peaks and abnormal azimuthal decay.
       - **Error Level Analysis (ELA):** Measures re-compression delta to spot spliced regions.
       - **Noise Residual Inconsistency:** Evaluates local block-wise variance across image regions.
       - **GLCM / LBP Texture:** Quantifies skin pore smoothing and unnatural homogeneity.
    5. **Explainable AI (Grad-CAM):** Visualizes high-gradient decision regions with contextual natural language forensic rationale.
    6. **Ensemble Layer & Uncertainty:** Combines all multi-stream forensic signals with dynamic weight rebalancing and calibrated uncertainty detection.
    7. **Feedback & Safe Retraining:** Continuous learning loop with strict validation safeguards against poisoned training samples.

    ---

    ### 📚 Public Research Datasets Guide
    To train or evaluate on large-scale public benchmark datasets:
    - **CIFAKE Dataset:** 120,000 Real vs AI-Generated (Diffusion/GAN) images.
    - **GenImage Benchmark:** Comprehensive benchmark covering Midjourney, Stable Diffusion, DALL-E, and VQ-GAN.
    - **FaceForensics++:** DeepFake, FaceSwap, Face2Face, and NeuralTextures video/image forensics.
    - **DiffusionDB:** Over 14 million prompt-image pairs from Stable Diffusion.
    - **Synthbuster:** Cross-generator synthetic dataset with FLUX, Midjourney v5/v6, and Firefly.
    """)
