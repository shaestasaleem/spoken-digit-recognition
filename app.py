import streamlit as st
import numpy as np
import librosa
import pickle
import os
import tempfile
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import load_model

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Spoken Digit Recognition",
    page_icon="🎤",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #f0f0f0;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

.result-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #00d2ff;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 30px rgba(0, 210, 255, 0.15);
}

.digit-display {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.confidence-text {
    font-size: 1.1rem;
    color: #aaa;
    margin-top: 0.5rem;
}

.model-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    margin: 0.3rem;
}

.svm-badge { background: rgba(0,210,255,0.15); border: 1px solid #00d2ff; color: #00d2ff; }
.cnn-badge { background: rgba(123,47,247,0.15); border: 1px solid #7b2ff7; color: #b48ffb; }

.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #00d2ff;
    border-bottom: 1px solid rgba(0,210,255,0.3);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

div[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
}

.stButton > button {
    background: linear-gradient(135deg, #00d2ff, #7b2ff7);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.6rem 2rem;
    width: 100%;
}

.stButton > button:hover {
    opacity: 0.85;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎤 Spoken Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speech Processing — Final Lab Project | Group #3</div>', unsafe_allow_html=True)


# ─── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        with open("svm_model.pkl", "rb") as f:
            models["svm"] = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            models["le"] = pickle.load(f)
        models["svm_loaded"] = True
    except:
        models["svm_loaded"] = False

    try:
        models["cnn"] = load_model("cnn_model.h5")
        models["cnn_loaded"] = True
    except:
        models["cnn_loaded"] = False

    return models

models = load_models()


# ─── Feature Extraction ────────────────────────────────────────
def extract_features_svm(signal, sr):
    signal_8k = librosa.resample(signal, orig_sr=sr, target_sr=8000)
    mfcc      = librosa.feature.mfcc(y=signal_8k, sr=8000, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    zcr       = np.mean(librosa.feature.zero_crossing_rate(signal_8k))
    energy    = np.mean(signal_8k ** 2)
    return np.hstack([mfcc_mean, zcr, energy])

def extract_spectrogram_cnn(signal, sr, img_h=64, img_w=64):
    signal_8k = librosa.resample(signal, orig_sr=sr, target_sr=8000)
    mel       = librosa.feature.melspectrogram(y=signal_8k, sr=8000, n_mels=img_h, fmax=4000)
    mel_db    = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < img_w:
        mel_db = np.pad(mel_db, ((0,0),(0, img_w - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :img_w]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db[..., np.newaxis][np.newaxis, ...]   # shape: (1, 64, 64, 1)


def predict_digit(signal, sr, model_choice):
    results = {}

    # SVM prediction
    if model_choice in ["SVM", "Both"] and models.get("svm_loaded"):
        feat    = extract_features_svm(signal, sr).reshape(1, -1)
        feat_sc = models["scaler"].transform(feat)
        pred    = models["svm"].predict(feat_sc)[0]
        label   = models["le"].inverse_transform([pred])[0]
        # SVM decision function for confidence
        scores  = models["svm"].decision_function(feat_sc)[0]
        conf    = round(float(np.max(scores) / np.sum(np.abs(scores)) * 100 + 50), 1)
        conf    = min(conf, 99.9)
        results["svm"] = {"digit": label, "confidence": conf}

    # CNN prediction
    if model_choice in ["CNN", "Both"] and models.get("cnn_loaded"):
        spec     = extract_spectrogram_cnn(signal, sr)
        probs    = models["cnn"].predict(spec, verbose=0)[0]
        digit    = str(np.argmax(probs))
        conf     = round(float(np.max(probs)) * 100, 1)
        results["cnn"] = {"digit": digit, "confidence": conf}

    return results


# ─── Model Status ──────────────────────────────────────────────
with st.expander("🔧 Model Status", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if models.get("svm_loaded"):
            st.success("✅ SVM Model loaded")
        else:
            st.error("❌ SVM Model not found\nRun notebook first to save models")
    with col2:
        if models.get("cnn_loaded"):
            st.success("✅ CNN Model loaded")
        else:
            st.error("❌ CNN Model not found\nRun notebook first to save models")


# ─── Model Selection ───────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Select Model</div>', unsafe_allow_html=True)
model_choice = st.radio(
    "Which model to use for prediction?",
    ["SVM", "CNN", "Both"],
    horizontal=True,
    index=2
)


# ─── Input Tabs ────────────────────────────────────────────────
st.markdown('<div class="section-header">🎧 Input Method</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📁  Upload Audio File", "🎙️  Record from Microphone"])


# ═══ TAB 1: File Upload ════════════════════════════════════════
with tab1:
    st.markdown('<div class="info-card">Upload a <b>.wav</b> or <b>.ogg</b> audio file of a spoken digit (0–9)</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose an audio file", type=["wav", "ogg", "mp3"])

    if uploaded is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Load and play audio
        signal, sr = librosa.load(tmp_path, sr=None)
        st.audio(tmp_path)

        st.markdown(f"**Sample Rate:** {sr} Hz &nbsp;|&nbsp; **Duration:** {round(len(signal)/sr, 2)} sec")

        if st.button("🔮 Predict Digit", key="predict_upload"):
            with st.spinner("Analyzing audio..."):
                results = predict_digit(signal, sr, model_choice)

            if results:
                if model_choice == "Both" and "svm" in results and "cnn" in results:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="model-badge svm-badge">SVM MODEL</span>
                            <div class="digit-display">{results['svm']['digit']}</div>
                            <div class="confidence-text">Confidence: {results['svm']['confidence']}%</div>
                        </div>""", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="model-badge cnn-badge">CNN MODEL</span>
                            <div class="digit-display">{results['cnn']['digit']}</div>
                            <div class="confidence-text">Confidence: {results['cnn']['confidence']}%</div>
                        </div>""", unsafe_allow_html=True)

                elif "svm" in results:
                    st.markdown(f"""
                    <div class="result-box">
                        <span class="model-badge svm-badge">SVM MODEL</span>
                        <div class="digit-display">{results['svm']['digit']}</div>
                        <div class="confidence-text">Confidence: {results['svm']['confidence']}%</div>
                    </div>""", unsafe_allow_html=True)

                elif "cnn" in results:
                    st.markdown(f"""
                    <div class="result-box">
                        <span class="model-badge cnn-badge">CNN MODEL</span>
                        <div class="digit-display">{results['cnn']['digit']}</div>
                        <div class="confidence-text">Confidence: {results['cnn']['confidence']}%</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Models not loaded. Please run the notebook first to train and save models.")

        os.unlink(tmp_path)


# ═══ TAB 2: Microphone Recording ══════════════════════════════
with tab2:
    st.markdown('<div class="info-card">Record yourself speaking a digit (0–9) using your microphone.</div>',
                unsafe_allow_html=True)

    duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=5, value=2)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔴 Start Recording", key="record"):
            with st.spinner(f"Recording for {duration} seconds..."):
                audio_data = sd.rec(
                    int(duration * 16000),
                    samplerate=16000,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                audio_data = audio_data.flatten()

            # Save recording
            rec_path = "recorded_digit.wav"
            sf.write(rec_path, audio_data, 16000)
            st.session_state["recorded_path"] = rec_path
            st.session_state["recorded_signal"] = audio_data
            st.session_state["recorded_sr"] = 16000
            st.success("✅ Recording complete!")

    with col2:
        if st.button("🔮 Predict Recorded", key="predict_mic"):
            if "recorded_signal" in st.session_state:
                with st.spinner("Analyzing recording..."):
                    results = predict_digit(
                        st.session_state["recorded_signal"],
                        st.session_state["recorded_sr"],
                        model_choice
                    )

                if "recorded_path" in st.session_state:
                    st.audio(st.session_state["recorded_path"])

                if results:
                    if model_choice == "Both" and "svm" in results and "cnn" in results:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"""
                            <div class="result-box">
                                <span class="model-badge svm-badge">SVM MODEL</span>
                                <div class="digit-display">{results['svm']['digit']}</div>
                                <div class="confidence-text">Confidence: {results['svm']['confidence']}%</div>
                            </div>""", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""
                            <div class="result-box">
                                <span class="model-badge cnn-badge">CNN MODEL</span>
                                <div class="digit-display">{results['cnn']['digit']}</div>
                                <div class="confidence-text">Confidence: {results['cnn']['confidence']}%</div>
                            </div>""", unsafe_allow_html=True)
                    elif "svm" in results:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="model-badge svm-badge">SVM MODEL</span>
                            <div class="digit-display">{results['svm']['digit']}</div>
                            <div class="confidence-text">Confidence: {results['svm']['confidence']}%</div>
                        </div>""", unsafe_allow_html=True)
                    elif "cnn" in results:
                        st.markdown(f"""
                        <div class="result-box">
                            <span class="model-badge cnn-badge">CNN MODEL</span>
                            <div class="digit-display">{results['cnn']['digit']}</div>
                            <div class="confidence-text">Confidence: {results['cnn']['confidence']}%</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Models not loaded. Please run the notebook first.")
            else:
                st.warning("⚠️ No recording found. Please record first!")


# ─── About Section ─────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    **Spoken Digit Recognition (0–9)** — Speech Processing Final Lab Project

    | | |
    |---|---|
    | **Dataset** | Free Spoken Digit Dataset (FSDD) — 3000 WAV files |
    | **Model 1** | SVM with MFCC + ZCR + Energy features |
    | **Model 2** | CNN with Mel Spectrogram images (64×64) |
    | **SVM Accuracy** | ~96.33% |

    **Group #3:** Savaira Majeed · Shaesta Saleem · Mustajab Zahra · Faizan Nazik · Sarfraz Ahmad · Muhammad Gulfam
    """)
