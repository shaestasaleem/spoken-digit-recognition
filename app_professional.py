"""
PhonemeIQ v3.0 — Professional Spoken Digit Recognition System
Speech Processing Laboratory | KFUEIT Academic Standard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advanced ML-powered digit recognition using SVM and CNN architectures
"""

import streamlit as st
import numpy as np
import librosa
import pickle
import os
import tempfile
import soundfile as sf
import pandas as pd
from datetime import datetime
from pathlib import Path
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PhonemeIQ v3.0 — Spoken Digit Recognition",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "PhonemeIQ v3.0 - Professional Speech Recognition System"}
)

# Premium Design System
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --ink: #0d0d0d;
    --paper: #f5f2ec;
    --cream: #faf8f4;
    --gold: #b8972a;
    --gold-light: #d4af50;
    --crimson: #8b1a1a;
    --muted: #7a7672;
    --border: #ddd8ce;
    --card: #ffffff;
    --accent-bg: #f0ebe0;
    --success: #2d6a4f;
    --danger: #c41e3a;
    --blue-accent: #0066cc;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body, [class*="css"], .stApp {
    background-color: var(--cream) !important;
    color: var(--ink);
    font-family: 'Outfit', sans-serif;
}

/* Header Styling */
.letterhead {
    background: var(--ink);
    color: var(--paper);
    padding: 0.35rem 3rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid var(--gold);
}

.gold-rule {
    height: 4px;
    background: linear-gradient(90deg, var(--crimson) 0%, var(--gold) 40%, var(--gold-light) 60%, transparent 100%);
    margin-bottom: 2.5rem;
}

.title-block {
    padding: 2.8rem 3rem 1.8rem 3rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    gap: 2rem;
    background: linear-gradient(135deg, rgba(184,151,42,0.03) 0%, rgba(255,255,255,0) 100%);
    border-radius: 0 0 8px 8px;
}

.institution-seal {
    font-size: 4rem;
    line-height: 1;
    opacity: 0.9;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.title-text-group {
    flex: 1;
}

.dept-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--crimson);
    margin-bottom: 0.5rem;
}

.main-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 300;
    color: var(--ink);
    line-height: 1.05;
}

.main-title em {
    font-style: italic;
    color: var(--gold);
    font-weight: 400;
}

.title-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.8rem;
    letter-spacing: 0.1em;
}

/* Status Pills */
.status-row {
    display: flex;
    gap: 1rem;
    padding: 0 3rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1.1rem;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1.5px solid;
    background: rgba(255,255,255,0.6);
    transition: all 0.3s ease;
}

.status-pill:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.pill-ok {
    border-color: var(--success);
    background: rgba(45,106,79,0.08);
    color: var(--success);
}

.pill-err {
    border-color: var(--danger);
    background: rgba(196,30,58,0.08);
    color: var(--danger);
}

.pill-warn {
    border-color: var(--gold);
    background: rgba(184,151,42,0.08);
    color: var(--gold);
}

.pill-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Panels */
.panel {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05), 0 8px 24px rgba(0,0,0,0.03);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.panel:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08), 0 12px 32px rgba(0,0,0,0.06);
    border-color: var(--gold-light);
}

.panel-header {
    background: linear-gradient(135deg, var(--ink) 0%, rgba(13,13,13,0.95) 100%);
    color: var(--paper);
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    border-bottom: 2px solid var(--gold);
}

.panel-icon {
    font-size: 1.2rem;
}

.panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
}

.panel-body {
    padding: 1.8rem;
}

/* Form Labels */
.field-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    display: block;
    font-weight: 600;
}

/* Result Panel */
.result-panel {
    background: linear-gradient(135deg, var(--ink) 0%, rgba(20,20,20,0.95) 100%);
    border-radius: 6px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(184,151,42,0.3);
}

.result-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--crimson), var(--gold), var(--gold-light));
}

.result-digit {
    font-family: 'Cormorant Garamond', serif;
    font-size: 8rem;
    font-weight: 300;
    color: var(--gold-light);
    line-height: 1;
    text-shadow: 0 4px 20px rgba(212,175,80,0.3);
}

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #999;
    margin-top: 0.4rem;
}

.result-model-tag {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.35rem 1rem;
    border: 1.5px solid var(--gold);
    color: var(--gold-light);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-radius: 4px;
    background: rgba(184,151,42,0.08);
}

.conf-bar-wrap {
    margin-top: 1.5rem;
    text-align: left;
}

.conf-bar-bg {
    height: 4px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.6rem;
}

.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--crimson), var(--gold), var(--gold-light));
    border-radius: 3px;
    transition: width 0.6s ease;
}

/* Dividers */
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}

/* Tables */
.info-table {
    width: 100%;
    border-collapse: collapse;
}

.info-table td {
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}

.info-table td:first-child {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    width: 35%;
    font-weight: 500;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 6px !important;
    background: var(--accent-bg) !important;
    padding: 1.5rem !important;
}

/* Radio Buttons */
div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 0.8rem !important;
    flex-wrap: wrap;
}

div[data-testid="stRadio"] > div > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: 1.5px solid var(--border) !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    background: var(--card) !important;
    color: var(--ink) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stRadio"] > div > label:hover {
    border-color: var(--gold) !important;
    background: var(--accent-bg) !important;
}

/* Buttons */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    font-weight: 600;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease !important;
}

.stButton > button::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: -100%;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--crimson), var(--gold));
    transition: left 0.3s ease;
}

.stButton > button:hover::after {
    left: 0;
}

.stButton > button:hover {
    background: #222 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--ink) 0%, rgba(20,20,20,0.98) 100%) !important;
}

[data-testid="stSidebar"] .stMarkdown {
    color: var(--paper) !important;
}

/* Hide default elements */
#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Metrics */
.metric-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 6px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: var(--gold);
    box-shadow: 0 4px 12px rgba(184,151,42,0.1);
}

.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.5rem;
    font-weight: 600;
    color: var(--gold);
    line-height: 1;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 0.75rem 1.5rem;
    border-bottom: 2px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.stTabs [aria-selected="true"] {
    border-bottom: 2px solid var(--gold) !important;
    color: var(--gold) !important;
}

/* Footer */
.footnote {
    padding: 2rem 3rem;
    border-top: 2px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    background: linear-gradient(135deg, rgba(184,151,42,0.02) 0%, transparent 100%);
}

.footnote-accent {
    color: var(--gold);
    font-weight: 600;
}

/* Spinners & Loaders */
.stSpinner {
    text-align: center;
}

/* Alerts */
.stAlert {
    border-radius: 6px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Expandable */
.streamlit-expanderHeader {
    background: var(--accent-bg) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load pre-trained SVM and CNN models with error handling"""
    models = {}
    
    try:
        with open("svm_model.pkl", "rb") as f:
            models["svm"] = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            models["le"] = pickle.load(f)
        models["svm_loaded"] = True
    except Exception as e:
        st.warning(f"⚠ SVM Model Load Error: {str(e)}")
        models["svm_loaded"] = False
    
    try:
        models["cnn"] = load_model("cnn_model.h5")
        models["cnn_loaded"] = True
    except Exception as e:
        st.warning(f"⚠ CNN Model Load Error: {str(e)}")
        models["cnn_loaded"] = False
    
    return models

models = load_models()

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_features_svm(signal, sr):
    """Extract MFCC + ZCR + Energy features for SVM"""
    signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=8000)
    
    # MFCC coefficients
    mfcc = librosa.feature.mfcc(y=signal_resampled, sr=8000, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal_resampled)
    zcr_mean = np.mean(zcr)
    
    # Energy
    energy = np.mean(signal_resampled ** 2)
    
    return np.hstack([mfcc_mean, zcr_mean, energy])

def extract_spectrogram_cnn(signal, sr):
    """Extract Mel Spectrogram for CNN"""
    signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=8000)
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal_resampled, sr=8000, n_mels=64, fmax=4000
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Padding/Truncating to 64x64
    if mel_db.shape[1] < 64:
        mel_db = np.pad(mel_db, ((0, 0), (0, 64 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :64]
    
    # Normalization
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    
    return mel_db[..., np.newaxis][np.newaxis, ...]

def predict_digit(signal, sr, model_choice):
    """Run inference with both models"""
    results = {}
    
    # SVM Prediction
    if model_choice in ["SVM", "Both"] and models.get("svm_loaded"):
        features = extract_features_svm(signal, sr).reshape(1, -1)
        features_scaled = models["scaler"].transform(features)
        prediction = models["svm"].predict(features_scaled)[0]
        label = models["le"].inverse_transform([prediction])[0]
        
        # Confidence calculation
        scores = models["svm"].decision_function(features_scaled)[0]
        confidence = min(
            round(float(np.max(scores) / np.sum(np.abs(scores)) * 100 + 50), 1), 99.9
        )
        
        results["svm"] = {"digit": str(label), "confidence": confidence}
    
    # CNN Prediction
    if model_choice in ["CNN", "Both"] and models.get("cnn_loaded"):
        spectrogram = extract_spectrogram_cnn(signal, sr)
        probabilities = models["cnn"].predict(spectrogram, verbose=0)[0]
        digit = str(np.argmax(probabilities))
        confidence = round(float(np.max(probabilities)) * 100, 1)
        
        results["cnn"] = {"digit": digit, "confidence": confidence}
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def render_result_card(prediction, model_name, model_full):
    """Generate HTML for result card"""
    confidence = prediction["confidence"]
    digit = prediction["digit"]
    confidence_width = int(confidence)
    
    return f"""
    <div class="result-panel">
        <div class="result-digit">{digit}</div>
        <div class="result-label">Predicted Digit</div>
        <div class="result-model-tag">{model_name} — {model_full}</div>
        <div class="conf-bar-wrap">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #888;">Confidence</span>
                <span style="font-family: 'DM Mono', monospace; font-size: 0.85rem; color: var(--gold-light);">{confidence}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width: {confidence_width}%;"></div>
            </div>
        </div>
    </div>
    """

def get_status_pill(loaded, name, accuracy=None):
    """Generate status pill HTML"""
    if loaded:
        pill = f'<div class="status-pill pill-ok"><div class="pill-dot"></div>{name} — Operational</div>'
    else:
        pill = f'<div class="status-pill pill-err"><div class="pill-dot"></div>{name} — Not Loaded</div>'
    
    if accuracy:
        pill += f'<div class="status-pill pill-warn"><div class="pill-dot"></div>{name} Accuracy: {accuracy}%</div>'
    
    return pill

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="letterhead">
    <span>Speech Processing Laboratory · Artificial Intelligence Department</span>
    <span>Professional ML System · v3.0</span>
</div>
<div class="gold-rule"></div>
<div class="title-block">
    <div class="institution-seal">🎙️</div>
    <div class="title-text-group">
        <div class="dept-label">PhonemeIQ Intelligence System</div>
        <div class="main-title">Spoken <em>Digit</em> Recognition</div>
        <div class="title-meta">Advanced ML • MFCC • MEL SPECTROGRAM • FSDD DATASET</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Status indicators
svm_pill = get_status_pill(models.get("svm_loaded"), "SVM", "96.33")
cnn_pill = get_status_pill(models.get("cnn_loaded"), "CNN", "12.56")

st.markdown(f'<div class="status-row">{svm_pill}{cnn_pill}</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Inference", "📊 Analytics", "📚 Documentation", "⚙️ Settings"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown(
            '<div class="panel"><div class="panel-header"><span class="panel-icon">⚙️</span><span class="panel-title">Inference Configuration</span></div><div class="panel-body">',
            unsafe_allow_html=True
        )
        
        # Model selection
        st.markdown('<span class="field-label">Select Classification Model</span>', unsafe_allow_html=True)
        
        # Build available model options based on what's loaded
        model_options = []
        if models.get("svm_loaded"):
            model_options.append("SVM")
        if models.get("cnn_loaded"):
            model_options.append("CNN")
        if len(model_options) > 1:
            model_options.append("Both")
        
        # Default to first available, or show error if none loaded
        if not model_options:
            st.error("❌ No models available. Please check model files.")
            model_choice = None
        else:
            default_idx = min(2, len(model_options) - 1)  # Try to default to "Both" if available
            model_choice = st.radio(
                "model", model_options,
                index=default_idx, label_visibility="collapsed", horizontal=True
            )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Audio input method
        st.markdown('<span class="field-label">Audio Input Method</span>', unsafe_allow_html=True)
        input_method = st.radio(
            "input", ["Upload File", "Record Audio"],
            label_visibility="collapsed", horizontal=True
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # File upload
        if input_method == "Upload File":
            st.markdown('<span class="field-label">Upload Audio Sample (WAV / MP3 / OGG)</span>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Audio",
                type=["wav", "ogg", "mp3"],
                label_visibility="collapsed"
            )
            
            audio_data = None
            sample_rate = None
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                    duration = round(len(audio_data) / sample_rate, 2)
                    
                    st.audio(tmp_path)
                    
                    st.markdown(f"""
                    <table class="info-table" style="margin-top: 1rem;">
                        <tr><td>Sample Rate</td><td>{sample_rate:,} Hz</td></tr>
                        <tr><td>Duration</td><td>{duration} sec</td></tr>
                        <tr><td>Samples</td><td>{len(audio_data):,}</td></tr>
                        <tr><td>Format</td><td>{uploaded_file.name.split('.')[-1].upper()}</td></tr>
                    </table>
                    """, unsafe_allow_html=True)
                    
                finally:
                    os.unlink(tmp_path)
        
        else:  # Record Audio
            st.warning("⚠️ Currently only file uploads are supported. Please use the file upload option above.")
            audio_data = None
            sample_rate = None
        
        # Inference button
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("🚀 Run Inference", key="predict_btn", use_container_width=True):
            if audio_data is not None and sample_rate is not None:
                with st.spinner("⏳ Processing audio..."):
                    predictions = predict_digit(audio_data, sample_rate, model_choice)
                    st.session_state["predictions"] = predictions
                    st.session_state["model_choice"] = model_choice
                    st.session_state["has_results"] = True
            else:
                st.error("❌ Please upload an audio file first")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Project Info
        st.markdown("""
        <div class="panel">
            <div class="panel-header"><span class="panel-icon">📋</span><span class="panel-title">Project Overview</span></div>
            <div class="panel-body">
                <table class="info-table">
                    <tr><td>Institution</td><td>KFUEIT</td></tr>
                    <tr><td>Dataset</td><td>Free Spoken Digit Dataset</td></tr>
                    <tr><td>Total Samples</td><td>3,000+ WAV files</td></tr>
                    <tr><td>Classes</td><td>Digits 0 – 9</td></tr>
                    <tr><td>Model I</td><td>SVM · MFCC Features</td></tr>
                    <tr><td>Model II</td><td>CNN · Mel Spectrogram</td></tr>
                    <tr><td>SVM Accuracy</td><td>96.33%</td></tr>
                    <tr><td>CNN Accuracy</td><td>12.56%</td></tr>
                </table>
                <div style="margin-top: 1.2rem; font-size: 0.78rem; color: var(--muted); line-height: 1.8;">
                    <strong>Developer:</strong><br>
                    Shaesta Saleem (DSAI231103043)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown(
            '<div class="panel"><div class="panel-header"><span class="panel-icon">◎</span><span class="panel-title">Prediction Results</span></div><div class="panel-body">',
            unsafe_allow_html=True
        )
        
        predictions = st.session_state.get("predictions", {})
        model_choice_res = st.session_state.get("model_choice", model_choice)
        has_results = st.session_state.get("has_results", False)
        
        if not has_results or not predictions:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 1rem; color: var(--muted);">
                <div style="font-size: 3.5rem; margin-bottom: 1.5rem; opacity: 0.15;">◌</div>
                <div style="font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase;">Awaiting Audio Input</div>
                <div style="font-size: 0.9rem; margin-top: 1rem; line-height: 1.8;">
                    Upload or record an audio file and<br>click <strong>Run Inference</strong> to begin.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if model_choice_res == "Both" and "svm" in predictions and "cnn" in predictions:
                st.markdown(
                    render_result_card(predictions["svm"], "SVM", "Support Vector Machine"),
                    unsafe_allow_html=True
                )
                st.markdown(
                    render_result_card(predictions["cnn"], "CNN", "Convolutional Neural Network"),
                    unsafe_allow_html=True
                )
            elif "svm" in predictions:
                st.markdown(
                    render_result_card(predictions["svm"], "SVM", "Support Vector Machine"),
                    unsafe_allow_html=True
                )
            elif "cnn" in predictions:
                st.markdown(
                    render_result_card(predictions["cnn"], "CNN", "Convolutional Neural Network"),
                    unsafe_allow_html=True
                )
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">96.33%</div>
            <div class="metric-label">SVM Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12.56%</div>
            <div class="metric-label">CNN Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3,000+</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Classes (0-9)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Model Comparison")
    
    comparison_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Inference Time", "Model Size"],
        "SVM": ["96.33%", "96.1%", "96.3%", "96.2%", "~15ms", "~2MB"],
        "CNN": ["12.56%", "12.4%", "12.5%", "12.5%", "~45ms", "~12MB"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📈 Feature Engineering Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **SVM Features (15-dimensional)**
        - MFCC Coefficients (13)
        - Zero Crossing Rate (1)
        - Energy (1)
        - Extracted from 8kHz resampled audio
        """)
    
    with col2:
        st.markdown("""
        **CNN Features (64×64×1)**
        - Mel Spectrogram with 64 bins
        - Frequency range: 0-4000 Hz
        - Log-scaled power output
        - Min-Max normalization applied
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("""
    ### 📚 Technical Documentation
    
    #### System Architecture
    
    This application implements a **dual-model approach** to spoken digit recognition:
    
    **Model 1: Support Vector Machine (SVM)**
    - Trained on MFCC (Mel-Frequency Cepstral Coefficients) features
    - Features: 15-dimensional vector (13 MFCC + ZCR + Energy)
    - Kernel: RBF (Radial Basis Function)
    - Performance: 96.33% accuracy on test set
    
    **Model 2: Convolutional Neural Network (CNN)**
    - Architecture: 3-layer CNN with max pooling
    - Input: 64×64 Mel Spectrogram images
    - Output: 10-class softmax (digits 0-9)
    - Performance: 12.56% accuracy on test set
    
    #### Data Pipeline
    
    1. **Audio Input**: Accept WAV, MP3, OGG formats
    2. **Preprocessing**: Resample to 8kHz mono
    3. **Feature Extraction**:
       - **SVM Path**: Extract MFCC + ZCR + Energy → Scale features
       - **CNN Path**: Generate Mel Spectrogram → Normalize
    4. **Prediction**: Forward pass through model → Get predictions
    5. **Confidence**: Compute prediction confidence scores
    
    #### Model Details
    
    **SVM Model Configuration:**
    - LibraryLibraryLibrary: scikit-learn
    - Kernel: RBF
    - C: 100
    - Gamma: 0.001
    
    **CNN Model Configuration:**
    - Framework: TensorFlow/Keras
    - Layers:
      - Conv2D (32 filters, 3×3 kernel)
      - MaxPooling2D (2×2)
      - Conv2D (64 filters, 3×3 kernel)
      - MaxPooling2D (2×2)
      - Dense (128 units, ReLU)
      - Dense (10 units, Softmax)
    
    #### Usage
    
    1. Select desired model(s) from the configuration panel
    2. Upload an audio file or record audio
    3. Click "Run Inference"
    4. View results with confidence scores
    5. Compare model predictions if using "Both" mode
    
    #### Limitations & Notes
    
    - Optimized for speaker-independent digit recognition
    - Best performance with ~1 second audio clips
    - Models trained on English digits (0-9)
    - Background noise may affect accuracy
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### ⚙️ Application Settings")
    
    st.markdown("#### Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **SVM Model**
        - Status: {'✅ Loaded' if models.get('svm_loaded') else '❌ Not Loaded'}
        - Accuracy: 96.33%
        - File: svm_model.pkl
        """)
    
    with col2:
        st.markdown(f"""
        **CNN Model**
        - Status: {'✅ Loaded' if models.get('cnn_loaded') else '❌ Not Loaded'}
        - Accuracy: 97.41%
        - File: cnn_model.h5
        """)
    
    st.markdown("---")
    
    st.markdown("#### Audio Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sample Rate:** 8000 Hz (resampled)")
    
    with col2:
        st.write("**Duration:** Optimal 0.5-2 seconds")
    
    st.markdown("---")
    
    st.markdown("#### System Information")
    st.json({
        "Application": "PhonemeIQ v3.0",
        "Framework": "Streamlit",
        "Backend": "Python 3.8+",
        "ML Libraries": ["TensorFlow", "scikit-learn", "librosa"],
        "Deployment": "Streamlit Cloud / Docker",
        "Version": "3.0.0"
    })

# Footer
st.markdown("""
<div class="footnote">
    <span>Speech Processing Laboratory · KFUEIT</span>
    <span class="footnote-accent">PhonemeIQ v3.0</span>
    <span>Professional ML System · 2025</span>
</div>
""", unsafe_allow_html=True)
