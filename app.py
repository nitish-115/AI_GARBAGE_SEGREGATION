# ============================================================
#  app.py — Streamlit Web Application
#  AI-Powered Garbage Segregation System
# ============================================================

import os
import sys
import json
import time
import base64
import tempfile
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

try:
    import cv2
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    CLASS_NAMES, BIN_MAP, CLASS_COLORS,
    MODEL_PATH, SAVED_MODEL_DIR, MODELS_DIR
)
from utils.dataset_utils import get_class_counts

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Garbage Segregation System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f1a;
    color: #e0e0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1123 0%, #1a1d35 100%);
    border-right: 1px solid #2a2d4a;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00e5ff 0%, #00ff88 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
}
.main-subtitle {
    color: #8888aa;
    font-size: 1rem;
    margin-top: 0.3rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d35 0%, #252848 100%);
    border: 1px solid #3a3d6a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2rem; font-weight: 700; color: #00e5ff; }
.metric-label { font-size: 0.8rem; color: #888; margin-top: 4px; }

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, #1a1d35, #252848);
    border-radius: 16px;
    padding: 1.5rem;
    border-left: 4px solid;
    margin: 1rem 0;
}
.waste-class {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
}
.bin-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
.confidence-bar-wrap { margin: 1rem 0; }
.conf-label { font-size: 0.8rem; color: #888; margin-bottom: 4px; }

/* Bin info box */
.bin-info {
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 0.8rem;
}

/* Feature cards */
.feature-card {
    background: #1a1d35;
    border: 1px solid #2a2d4a;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 0.8rem;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.3rem; }
.feature-desc  { color: #888; font-size: 0.85rem; line-height: 1.5; }

/* Section header */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00e5ff;
    border-bottom: 1px solid #2a2d4a;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem;
    letter-spacing: 1px;
}

/* Class badge grid */
.badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
         font-size: 0.8rem; margin: 0.2rem; font-weight: 500; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: #2a2d4a; border-radius: 3px; }

/* Streamlit overrides */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #00e5ff, #00b4d8);
    color: #0d0f1a;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #00ff88, #00e5ff);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
def model_trained() -> bool:
    return os.path.exists(MODEL_PATH)

def load_metrics() -> dict:
    p = os.path.join(SAVED_MODEL_DIR, "metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

def load_history() -> dict:
    p = os.path.join(SAVED_MODEL_DIR, "training_history.npy")
    if os.path.exists(p):
        return np.load(p, allow_pickle=True).item()
    return {}

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_resource
def get_predictor():
    from predict import predict as _predict, load_model_and_classes
    load_model_and_classes()
    return _predict


def _get_disposal_tip(cls: str) -> str:
    tips = {
        "Battery":         "⚠️ Never put batteries in regular bins — they contain hazardous chemicals.",
        "Cardboard":       "Flatten boxes before placing them in the recycling bin.",
        "E-Waste":         "Drop off at an authorised e-waste collection point.",
        "Glass":           "Rinse before recycling. Broken glass needs special wrapping.",
        "Metal":           "Rinse food tins. Aluminium cans are highly recyclable.",
        "Organic Waste":   "Compost or place in the green organic waste bin.",
        "Paper":           "Keep dry — wet paper is not recyclable.",
        "Plastic":         "Check the resin code. Rinse containers before recycling.",
        "Textile/Clothes": "Donate wearable items. Worn-out textiles go to textile banks.",
        "Trash":           "Ensure non-recyclable waste is sealed before disposal.",
    }
    return tips.get(cls, "")


def render_prediction(img: Image.Image):
    """
    Shared rendering logic: runs the model on a PIL image and draws the
    result card + top-5 chart + disposal tip. Used by both the Upload
    page and the Live Camera page so behaviour stays identical.
    """
    with st.spinner("🔍 Analysing waste …"):
        predict_fn = get_predictor()
        result = predict_fn(img)

    cls       = result["class_name"]
    conf      = result["confidence"]
    bin_label = result["bin_label"]
    color     = result["bin_color"]

    st.markdown(f"""
    <div class='result-card' style='border-color:{color};'>
      <div style='font-size:0.8rem; color:#888; margin-bottom:4px;'>Detected Waste</div>
      <div class='waste-class' style='color:{color};'>{cls}</div>
      <div class='bin-badge' style='background:{color}22; border:1px solid {color}; color:{color};'>
        {bin_label}
      </div>
      <div class='confidence-bar-wrap'>
        <div class='conf-label'>Confidence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(conf), text=f"**{conf*100:.1f}%** confidence")

    # Top-5 chart
    st.markdown("<div class='section-header'>TOP-5 PREDICTIONS</div>",
                unsafe_allow_html=True)
    top5_classes = [x[0] for x in result["top5"]]
    top5_probs   = [x[1]*100 for x in result["top5"]]
    colors_top5  = [CLASS_COLORS[CLASS_NAMES.index(c) % len(CLASS_COLORS)]
                    if c in CLASS_NAMES else "#607D8B"
                    for c in top5_classes]
    fig = go.Figure(go.Bar(
        x=top5_probs, y=top5_classes,
        orientation="h",
        marker_color=colors_top5,
        text=[f"{p:.1f}%" for p in top5_probs],
        textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a",
        height=240, margin=dict(l=10, r=60, t=10, b=10),
        font_color="#e0e0f0",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bin info
    st.markdown(f"""
    <div class='bin-info'>
      <b style='color:{color};'>♻️ Disposal Instruction</b><br>
      <span style='font-size:0.9rem;'>
        This item is classified as <b>{cls}</b>.<br>
        Place it in the <b>{bin_label}</b>.<br>
        {_get_disposal_tip(cls)}
      </span>
    </div>""", unsafe_allow_html=True)


# ── Live camera overlay helpers (ported from webcam_detection.py) ──
_HEX_CACHE: dict = {}

def hex_to_bgr(hex_color: str) -> tuple:
    if hex_color in _HEX_CACHE:
        return _HEX_CACHE[hex_color]
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    bgr = (b, g, r)
    _HEX_CACHE[hex_color] = bgr
    return bgr


def draw_info_panel(frame, result: dict, fps: float, h: int, w: int):
    panel_w = 340
    panel_h = 260
    px, py = 15, 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    bin_color = hex_to_bgr(result["bin_color"])
    cls_name  = result["class_name"]
    conf      = result["confidence"] * 100
    bin_label = result["bin_label"]

    cv2.rectangle(frame, (px, py), (px + panel_w, py + 36), bin_color, -1)
    cv2.putText(frame, "AI GARBAGE DETECTOR", (px + 8, py + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, cls_name, (px + 8, py + 64),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, bin_color, 2, cv2.LINE_AA)

    bar_x, bar_y = px + 8, py + 84
    bar_w = panel_w - 16
    bar_fill = int(bar_w * result["confidence"])
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (50, 50, 70), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 14), bin_color, -1)
    cv2.putText(frame, f"{conf:.1f}%", (bar_x + bar_w - 52, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Confidence", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "Disposal Bin:", (px + 8, py + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, bin_label.split("  ")[-1] if "  " in bin_label else bin_label,
                (px + 8, py + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Top Predictions:", (px + 8, py + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 200), 1, cv2.LINE_AA)
    for i, (cls, prob) in enumerate(result["top5"][:3]):
        ty = py + 183 + i * 22
        mini_fill = int((panel_w - 16) * 0.6 * prob)
        cv2.rectangle(frame, (px + 8, ty), (px + 8 + mini_fill, ty + 10),
                      bin_color if i == 0 else (80, 80, 100), -1)
        cv2.putText(frame, f"{cls} {prob*100:.0f}%", (px + 8, ty + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 90, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)


if WEBRTC_AVAILABLE:
    class LiveDetectionProcessor(VideoProcessorBase):
        """
        Runs on every incoming camera frame inside the browser stream.
        Predicts every `predict_every` frames (throttled for speed) and
        draws the same overlay panel the old OpenCV window used to show.
        """
        def __init__(self):
            self.predict_fn    = get_predictor()
            self.predict_every = 15
            self.roi_fraction  = 0.5   # detection box = this fraction of the shorter frame side
            self.frame_n       = 0
            self.result        = None
            self.prev_time     = time.time()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            self.frame_n += 1

            # Centered square detection box
            side = int(min(h, w) * self.roi_fraction)
            cx, cy = w // 2, h // 2
            x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
            x2, y2 = min(x1 + side, w), min(y1 + side, h)

            # Only classify what's INSIDE the box, not the whole frame
            if self.frame_n % self.predict_every == 1 or self.result is None:
                try:
                    roi = img[y1:y2, x1:x2]
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    self.result = self.predict_fn(pil_img)
                except Exception:
                    pass  # keep showing the last good result rather than crashing the stream

            now = time.time()
            fps = 1.0 / max(now - self.prev_time, 1e-9)
            self.prev_time = now

            # Dim everything outside the box so it's obvious where to hold the item
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            dimmed = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
            img = np.where(mask[..., None] == 255, img, dimmed).astype(np.uint8)

            box_color = hex_to_bgr(self.result["bin_color"]) if self.result else (255, 229, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(img, "Place item in box", (x1, max(y1 - 12, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if self.result:
                draw_info_panel(img, self.result, fps, h, w)

            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
          <div style='font-family: Orbitron, monospace; font-size: 1.1rem;
                      font-weight: 700; color: #00e5ff; letter-spacing:2px;'>
            ♻️ GarbageAI
          </div>
          <div style='color:#555; font-size:0.75rem; margin-top:2px;'>
            Intelligent Waste Classifier
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📤 Upload & Predict", "📷 Live Camera",
             "📊 Model Performance", "ℹ️ About Project"],
            label_visibility="collapsed"
        )
        st.markdown("---")

        # Model status
        if model_trained():
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model Not Trained")
            if st.button("🚀 Train Model Now"):
                st.session_state["run_training"] = True

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75rem; color:#555; text-align:center;'>
          Built with TensorFlow & MobileNetV2<br>
          © 2025 AI Garbage Segregation
        </div>
        """, unsafe_allow_html=True)

    return page


# ── Pages ─────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class='main-header'>
      <div class='main-title'>AI GARBAGE SEGREGATION SYSTEM</div>
      <div class='main-subtitle'>
        Computer Vision · Transfer Learning · Real-Time Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    counts = get_class_counts()
    total  = sum(counts.values())
    metrics = load_metrics()

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (str(len(CLASS_NAMES)), "Waste Categories"),
        (str(total) if total else "—", "Total Images"),
        (f"{metrics.get('accuracy', 0)*100:.1f}%" if metrics else "—", "Model Accuracy"),
        ("MobileNetV2", "Architecture"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown("<div class='section-header'>CORE FEATURES</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    features = [
        ("🖼️", "Image Upload", "Upload any waste image for instant AI classification with confidence scores."),
        ("📷", "Live Camera", "Point your laptop camera at an item and classify it instantly, right in the browser."),
        ("♻️", "Bin Routing", "Auto-assigns waste to the correct disposal bin with colour-coded guidance."),
        ("🧠", "Transfer Learning", "MobileNetV2 backbone fine-tuned on garbage data for high accuracy."),
        ("📈", "Analytics", "Live training curves, confusion matrix, and per-class performance metrics."),
        ("🔍", "10 Categories", "Classifies Plastic, Glass, Metal, Paper, E-Waste, Battery, and more."),
    ]
    for col, (icon, title, desc) in zip([r1, r2, r3, r1, r2, r3], features):
        col.markdown(f"""
        <div class='feature-card'>
          <div class='feature-icon'>{icon}</div>
          <div class='feature-title'>{title}</div>
          <div class='feature-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Waste classes
    st.markdown("<div class='section-header'>SUPPORTED WASTE CATEGORIES</div>",
                unsafe_allow_html=True)
    badge_html = ""
    for i, cls in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        lbl, _, icon = BIN_MAP.get(cls, ("General", "#607D8B", "⚫"))
        badge_html += f"<span class='badge' style='background:{color}22; border:1px solid {color}; color:{color};'>{icon} {cls}</span>"
    st.markdown(badge_html, unsafe_allow_html=True)

    # Dataset distribution chart
    if counts:
        st.markdown("<div class='section-header'>DATASET DISTRIBUTION</div>",
                    unsafe_allow_html=True)
        df = pd.DataFrame({"Category": list(counts.keys()), "Images": list(counts.values())})
        fig = px.bar(df, x="Category", y="Images", color="Category",
                     color_discrete_sequence=CLASS_COLORS[:len(counts)],
                     template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a",
            showlegend=False, margin=dict(t=20, b=40),
            height=320, font_color="#e0e0f0"
        )
        st.plotly_chart(fig, use_container_width=True)


def page_upload():
    st.markdown("<div class='main-title' style='font-size:1.6rem;'>📤 Upload & Predict</div>",
                unsafe_allow_html=True)
    st.markdown("Upload a waste image to classify it and get bin recommendations.")
    st.markdown("---")

    if not model_trained():
        st.error("Model not trained yet. Go to the sidebar and click **Train Model Now**.")
        return

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported: JPG, PNG, BMP, WEBP"
    )

    if uploaded:
        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with col_res:
            render_prediction(img)

def page_camera():
    st.markdown("<div class='main-title' style='font-size:1.6rem;'>📷 Live Camera Detection</div>",
                unsafe_allow_html=True)
    st.markdown(
        "Continuous real-time detection, right inside the browser — point an item at your "
        "camera and the class, confidence, and top-3 predictions update automatically."
    )
    st.markdown("---")

    if not model_trained():
        st.error("Model not trained yet. Train the model first.")
        return

    if not WEBRTC_AVAILABLE:
        st.error(
            "`streamlit-webrtc` isn't installed. Run this in your project's terminal, "
            "then restart the app:"
        )
        st.code("pip install streamlit-webrtc av", language="bash")
        return

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        predict_every = st.slider(
            "Prediction frequency (lower = faster updates, more CPU load)",
            min_value=5, max_value=30, value=15, step=5,
            help="Runs a prediction once every N frames instead of every single frame."
        )
    with col_s2:
        roi_fraction = st.slider(
            "Detection box size",
            min_value=0.3, max_value=0.9, value=0.5, step=0.1,
            help="Only the area inside the box is classified — everything outside is ignored."
        )

    ctx = webrtc_streamer(
        key="live-garbage-detection",
        video_processor_factory=LiveDetectionProcessor,
        rtc_configuration=RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:openrelay.metered.ca:80"]},
                    {
                        "urls": ["turn:openrelay.metered.ca:80"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                    {
                        "urls": ["turn:openrelay.metered.ca:443"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                    {
                        "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                ]
            }
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.predict_every = predict_every
        ctx.video_processor.roi_fraction  = roi_fraction

    st.caption(
        "Click **START** above, allow camera access in your browser, then hold "
        "a waste item **inside the highlighted box** (the area outside is dimmed "
        "and ignored by the model). Click **STOP** when done."
    )

     


  




def page_performance():
    st.markdown("<div class='main-title' style='font-size:1.6rem;'>📊 Model Performance</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    metrics = load_metrics()
    history = load_history()

    if not metrics:
        st.warning("No evaluation metrics found. Train the model first.")
        return

    # Metrics row
    m_cols = st.columns(4)
    m_data = [
        ("Accuracy",  metrics.get("accuracy",  0)),
        ("Precision", metrics.get("precision", 0)),
        ("Recall",    metrics.get("recall",    0)),
        ("F1 Score",  metrics.get("f1",        0)),
    ]
    colors_m = ["#00e5ff", "#00ff88", "#ff9800", "#e91e63"]
    for col, (name, val), color in zip(m_cols, m_data, colors_m):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:{color};'>{val*100:.1f}%</div>
          <div class='metric-label'>{name}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training curves
    if history:
        st.markdown("<div class='section-header'>TRAINING CURVES</div>",
                    unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        epochs = list(range(1, len(history.get("accuracy", [])) + 1))

        with tc1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history.get("accuracy", []),
                                     name="Train", line=dict(color="#4CAF50", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=history.get("val_accuracy", []),
                                     name="Validation", line=dict(color="#FF9800", width=2, dash="dash")))
            fig.update_layout(title="Accuracy", paper_bgcolor="#0d0f1a",
                              plot_bgcolor="#0d0f1a", font_color="#e0e0f0",
                              height=280, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with tc2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history.get("loss", []),
                                     name="Train", line=dict(color="#F44336", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=history.get("val_loss", []),
                                     name="Validation", line=dict(color="#2196F3", width=2, dash="dash")))
            fig.update_layout(title="Loss", paper_bgcolor="#0d0f1a",
                              plot_bgcolor="#0d0f1a", font_color="#e0e0f0",
                              height=280, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix image
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.markdown("<div class='section-header'>CONFUSION MATRIX</div>",
                    unsafe_allow_html=True)
        st.image(cm_path, use_column_width=True)

    # Bin breakdown table
    st.markdown("<div class='section-header'>BIN ROUTING TABLE</div>",
                unsafe_allow_html=True)
    bin_data = []
    for cls in CLASS_NAMES:
        lbl, color, icon = BIN_MAP.get(cls, ("General Waste", "#607D8B", "⚫"))
        bin_data.append({"Waste Category": f"{icon} {cls}", "Disposal Bin": lbl})
    st.dataframe(pd.DataFrame(bin_data), use_container_width=True, hide_index=True)


def page_about():
    st.markdown("<div class='main-title' style='font-size:1.6rem;'>ℹ️ About the Project</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ## 🌍 Project Overview

    This **AI-Powered Garbage Segregation System** uses computer vision and deep learning to
    automatically classify waste materials into 10 categories, guiding disposal into the
    correct bin. It runs entirely on a standard laptop — no external hardware required.

    ---

    ## 🏗️ System Architecture

    ```
    Raw Image Input
         │
         ▼
    ┌─────────────────────────┐
    │  Preprocessing Pipeline │  ← Resize 224×224, Normalise, Augment
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  MobileNetV2 Backbone   │  ← Pretrained on ImageNet (frozen)
    │  (Feature Extractor)    │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Custom Head            │  ← GlobalAvgPool → BN → Dense(512) → Dropout
    │                         │    → Dense(256) → Dropout → Softmax(10)
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Classification Result  │  ← Class + Confidence
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Bin Routing Engine     │  ← Maps class → disposal bin
    └─────────────────────────┘
    ```

    ---

    ## 🔧 Technology Stack

    | Component | Technology |
    |-----------|-----------|
    | Deep Learning | TensorFlow 2.x / Keras |
    | Base Model | MobileNetV2 (ImageNet weights) |
    | Computer Vision | OpenCV |
    | Web Interface | Streamlit |
    | Data Processing | NumPy, Pandas |
    | Visualisation | Matplotlib, Seaborn, Plotly |
    | ML Utilities | Scikit-learn |

    ---

    ## 📁 Project Structure

    ```
    garbage_segregation/
    ├── dataset/               ← Waste images by category subfolder
    ├── models/                ← Plots: confusion matrix, curves
    ├── saved_model/           ← Trained .h5 model + metrics JSON
    ├── utils/
    │   ├── config.py          ← Constants, class names, bin mappings
    │   └── dataset_utils.py   ← Dataset scanning & synthetic generation
    ├── train_preprocess.py    ← Data loading, augmentation, tf.data pipelines
    ├── train.py               ← Model definition, training, evaluation
    ├── predict.py             ← Inference engine (PIL / NumPy / path input)
    ├── webcam_detection.py    ← Real-time OpenCV detection loop (standalone)
    ├── app.py                 ← Streamlit multi-page application (incl. in-browser camera)
    ├── requirements.txt       ← Python dependencies
    └── README.md              ← Setup & usage guide
    ```

    ---

    ## 🚀 Training Strategy

    **Phase 1 — Head Training (frozen base)**
    MobileNetV2 weights are frozen; only the custom dense layers are trained.
    This allows the classifier to quickly learn meaningful features.

    **Phase 2 — Fine-Tuning (last 30 base layers unfrozen)**
    The top layers of MobileNetV2 are unfrozen and trained at a 10× lower
    learning rate, allowing domain adaptation to waste images.

    **Regularisation:** Dropout (0.4 / 0.2), Batch Normalisation,
    Early Stopping, ReduceLROnPlateau, and Model Checkpointing.
    """)


# ── Training runner ───────────────────────────────────────────
def run_training_ui():
    st.markdown("### 🚀 Training in Progress …")
    placeholder = st.empty()
    bar = st.progress(0, text="Initialising …")
    log_area = st.empty()

    import subprocess
    proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "train.py")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=os.path.dirname(__file__)
    )
    logs = []
    i = 0
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            logs.append(line.rstrip())
            log_area.code("\n".join(logs[-30:]), language="")
        i = min(i + 1, 95)
        bar.progress(i, text="Training … please wait")
        time.sleep(0.05)

    bar.progress(100, text="✅ Training complete!")
    if proc.returncode == 0:
        st.success("Model trained and saved successfully! Refresh the page.")
    else:
        st.error("Training failed. Check logs above.")
    st.session_state.pop("run_training", None)


# ── Main ──────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    if st.session_state.get("run_training"):
        run_training_ui()
        return

    if page == "🏠 Home":
        page_home()
    elif page == "📤 Upload & Predict":
        page_upload()
    elif page == "📷 Live Camera":
        page_camera()
    elif page == "📊 Model Performance":
        page_performance()
    elif page == "ℹ️ About Project":
        page_about()


if __name__ == "__main__":
    main()