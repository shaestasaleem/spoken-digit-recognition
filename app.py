import streamlit as st
import numpy as np
import librosa
import pickle
import os
import tempfile
import soundfile as sf
from keras.models import load_model

st.set_page_config(page_title="PhonemeIQ — Spoken Digit Recognition", page_icon="🎙", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');
:root{--ink:#0d0d0d;--paper:#f5f2ec;--cream:#faf8f4;--gold:#b8972a;--gold-light:#d4af50;--crimson:#8b1a1a;--muted:#7a7672;--border:#ddd8ce;--card:#ffffff;--accent-bg:#f0ebe0}
*{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{background-color:var(--cream)!important;color:var(--ink);font-family:'Outfit',sans-serif}
.letterhead{background:var(--ink);color:var(--paper);padding:.35rem 3rem;font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;display:flex;justify-content:space-between;align-items:center}
.gold-rule{height:3px;background:linear-gradient(90deg,var(--crimson) 0%,var(--gold) 40%,var(--gold-light) 60%,transparent 100%);margin-bottom:2.5rem}
.title-block{padding:2.8rem 3rem 1.8rem 3rem;border-bottom:1px solid var(--border);margin-bottom:2rem;display:flex;align-items:flex-end;gap:2rem}
.institution-seal{font-size:4rem;line-height:1;opacity:.9}
.title-text-group{flex:1}
.dept-label{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.22em;text-transform:uppercase;color:var(--crimson);margin-bottom:.5rem}
.main-title{font-family:'Cormorant Garamond',serif;font-size:3.4rem;font-weight:300;color:var(--ink);line-height:1.05}
.main-title em{font-style:italic;color:var(--gold)}
.title-meta{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted);margin-top:.8rem;letter-spacing:.1em}
.status-row{display:flex;gap:1rem;padding:0 3rem;margin-bottom:2rem;flex-wrap:wrap}
.status-pill{display:flex;align-items:center;gap:.5rem;padding:.45rem 1.1rem;border-radius:2px;font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;border:1px solid}
.pill-ok{border-color:#2d6a4f;background:#f0faf4;color:#2d6a4f}
.pill-err{border-color:var(--crimson);background:#fdf0f0;color:var(--crimson)}
.pill-dot{width:6px;height:6px;border-radius:50%;background:currentColor}
.panel{background:var(--card);border:1px solid var(--border);border-radius:2px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.03);margin-bottom:1.5rem}
.panel-header{background:var(--ink);color:var(--paper);padding:.9rem 1.5rem;display:flex;align-items:center;gap:.8rem}
.panel-icon{font-size:1rem}
.panel-title{font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase}
.panel-body{padding:1.8rem}
.field-label{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;display:block}
.result-panel{background:var(--ink);border-radius:2px;padding:2.5rem 2rem;text-align:center;position:relative;overflow:hidden;margin-bottom:1.2rem}
.result-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--crimson),var(--gold),var(--gold-light))}
.result-digit{font-family:'Cormorant Garamond',serif;font-size:7rem;font-weight:300;color:var(--gold-light);line-height:1}
.result-label{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:#888;margin-top:.4rem}
.result-model-tag{display:inline-block;margin-top:.8rem;padding:.25rem .8rem;border:1px solid var(--gold);color:var(--gold);font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;border-radius:1px}
.conf-bar-wrap{margin-top:1.2rem;text-align:left}
.conf-bar-bg{height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;margin-top:.4rem}
.conf-bar-fill{height:100%;background:linear-gradient(90deg,var(--crimson),var(--gold));border-radius:2px}
.divider{height:1px;background:var(--border);margin:1.5rem 0}
.info-table{width:100%;border-collapse:collapse}
.info-table td{padding:.6rem 0;border-bottom:1px solid var(--border);font-size:.85rem}
.info-table td:first-child{font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);width:40%}
div[data-testid="stFileUploader"]{border:1.5px dashed var(--border)!important;border-radius:2px!important;background:var(--accent-bg)!important;padding:1.2rem!important}
div[data-testid="stRadio"]>label{display:none}
div[data-testid="stRadio"]>div{display:flex!important;gap:.8rem!important;flex-wrap:wrap}
div[data-testid="stRadio"]>div>label{font-family:'DM Mono',monospace!important;font-size:.72rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;border:1px solid var(--border)!important;padding:.5rem 1.2rem!important;border-radius:2px!important;cursor:pointer!important;background:var(--card)!important;color:var(--ink)!important}
.stButton>button{background:var(--ink)!important;color:var(--paper)!important;border:none!important;border-radius:2px!important;font-family:'DM Mono',monospace!important;font-size:.72rem!important;letter-spacing:.15em!important;text-transform:uppercase!important;padding:.75rem 2rem!important;width:100%!important;position:relative;overflow:hidden}
.stButton>button::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--crimson),var(--gold))}
.stButton>button:hover{background:#222!important;opacity:1!important}
[data-testid="stSidebar"]{display:none}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0!important;max-width:100%!important}
.footnote{padding:1.5rem 3rem;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase}
.footnote-accent{color:var(--gold)}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    m = {}
    try:
        with open("svm_model.pkl","rb") as f: m["svm"]=pickle.load(f)
        with open("scaler.pkl","rb") as f:    m["scaler"]=pickle.load(f)
        with open("label_encoder.pkl","rb") as f: m["le"]=pickle.load(f)
        m["svm_loaded"]=True
    except: m["svm_loaded"]=False
    try:
        m["cnn"]=load_model("cnn_model.h5"); m["cnn_loaded"]=True
    except: m["cnn_loaded"]=False
    return m

models = load_models()

def extract_features_svm(signal, sr):
    s8=librosa.resample(signal,orig_sr=sr,target_sr=8000)
    mfcc=librosa.feature.mfcc(y=s8,sr=8000,n_mfcc=13)
    return np.hstack([np.mean(mfcc,axis=1), np.mean(librosa.feature.zero_crossing_rate(s8)), np.mean(s8**2)])

def extract_spectrogram_cnn(signal, sr):
    s8=librosa.resample(signal,orig_sr=sr,target_sr=8000)
    mel=librosa.feature.melspectrogram(y=s8,sr=8000,n_mels=64,fmax=4000)
    mel_db=librosa.power_to_db(mel,ref=np.max)
    if mel_db.shape[1]<64: mel_db=np.pad(mel_db,((0,0),(0,64-mel_db.shape[1])))
    else: mel_db=mel_db[:,:64]
    mel_db=(mel_db-mel_db.min())/(mel_db.max()-mel_db.min()+1e-6)
    return mel_db[...,np.newaxis][np.newaxis,...]

def predict_digit(signal, sr, model_choice):
    results={}
    if model_choice in ["SVM","Both"] and models.get("svm_loaded"):
        feat=extract_features_svm(signal,sr).reshape(1,-1)
        feat_sc=models["scaler"].transform(feat)
        pred=models["svm"].predict(feat_sc)[0]
        label=models["le"].inverse_transform([pred])[0]
        scores=models["svm"].decision_function(feat_sc)[0]
        conf=min(round(float(np.max(scores)/np.sum(np.abs(scores))*100+50),1),99.9)
        results["svm"]={"digit":label,"confidence":conf}
    if model_choice in ["CNN","Both"] and models.get("cnn_loaded"):
        probs=models["cnn"].predict(extract_spectrogram_cnn(signal,sr),verbose=0)[0]
        results["cnn"]={"digit":str(np.argmax(probs)),"confidence":round(float(np.max(probs))*100,1)}
    return results

def result_card(r, tag, full):
    bw=int(r['confidence'])
    return f"""<div class="result-panel">
        <div class="result-digit">{r['digit']}</div>
        <div class="result-label">Predicted Digit</div>
        <div class="result-model-tag">{tag} — {full}</div>
        <div class="conf-bar-wrap">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#666">Confidence</span>
                <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:#d4af50">{r['confidence']}%</span>
            </div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bw}%"></div></div>
        </div>
    </div>"""

# ── Render ──────────────────────────────────────────────────────
st.markdown("""
<div class="letterhead">
    <span>Dept. of Electrical Engineering · Speech Processing Lab</span>
    <span>Final Laboratory Project · Group III</span>
</div>
<div class="gold-rule"></div>
<div class="title-block">
    <div class="institution-seal">🎙</div>
    <div class="title-text-group">
        <div class="dept-label">Phoneme Intelligence System · v2.0</div>
        <div class="main-title">Spoken <em>Digit</em> Recognition</div>
        <div class="title-meta">SVM · CNN · MFCC · MEL SPECTROGRAM · FSDD DATASET</div>
    </div>
</div>
""", unsafe_allow_html=True)

sp = '<div class="status-pill pill-ok"><div class="pill-dot"></div>SVM — Operational</div>' if models.get("svm_loaded") else '<div class="status-pill pill-err"><div class="pill-dot"></div>SVM — Not Loaded</div>'
cp = '<div class="status-pill pill-ok"><div class="pill-dot"></div>CNN — Operational</div>' if models.get("cnn_loaded") else '<div class="status-pill pill-err"><div class="pill-dot"></div>CNN — Not Loaded</div>'
st.markdown(f'<div class="status-row">{sp}{cp}<div class="status-pill" style="border-color:#b8972a;background:#fdf9ee;color:#b8972a;"><div class="pill-dot"></div>SVM Acc: 96.33%</div></div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="panel"><div class="panel-header"><span class="panel-icon">⚙</span><span class="panel-title">Inference Configuration</span></div><div class="panel-body">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Select Classification Model</span>', unsafe_allow_html=True)
    model_choice = st.radio("model", ["SVM","CNN","Both"], index=2, label_visibility="collapsed", horizontal=True)
    st.markdown('<div class="divider"></div><span class="field-label">Upload Audio Sample</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Audio", type=["wav","ogg","mp3"], label_visibility="collapsed")

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read()); tmp_path=tmp.name
        signal, sr = librosa.load(tmp_path, sr=None)
        duration   = round(len(signal)/sr, 2)
        st.audio(tmp_path)
        st.markdown(f"""<table class="info-table" style="margin-top:1rem">
            <tr><td>Sample Rate</td><td>{sr:,} Hz</td></tr>
            <tr><td>Duration</td><td>{duration} sec</td></tr>
            <tr><td>Samples</td><td>{len(signal):,}</td></tr>
            <tr><td>Format</td><td>{uploaded.name.split('.')[-1].upper()}</td></tr>
        </table><br>""", unsafe_allow_html=True)
        if st.button("Run Inference →", key="predict_upload"):
            with st.spinner("Processing audio..."):
                st.session_state["results"]      = predict_digit(signal, sr, model_choice)
                st.session_state["model_choice"] = model_choice
        os.unlink(tmp_path)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("""<div class="panel">
        <div class="panel-header"><span class="panel-icon">📋</span><span class="panel-title">Project Synopsis</span></div>
        <div class="panel-body">
            <table class="info-table">
                <tr><td>Dataset</td><td>Free Spoken Digit Dataset</td></tr>
                <tr><td>Samples</td><td>3,000 WAV recordings</td></tr>
                <tr><td>Classes</td><td>Digits 0 – 9</td></tr>
                <tr><td>Model I</td><td>SVM · MFCC + ZCR + Energy</td></tr>
                <tr><td>Model II</td><td>CNN · Mel Spectrogram 64×64</td></tr>
                <tr><td>SVM Accuracy</td><td>96.33 %</td></tr>
            </table>
            <div style="margin-top:1.2rem;font-size:.78rem;color:var(--muted);line-height:1.8">
                Savaira Majeed &nbsp;·&nbsp; Shaesta Saleem &nbsp;·&nbsp; Mustajab Zahra<br>
                Faizan Nazik &nbsp;·&nbsp; Sarfraz Ahmad &nbsp;·&nbsp; Muhammad Gulfam
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel"><div class="panel-header"><span class="panel-icon">◎</span><span class="panel-title">Prediction Output</span></div><div class="panel-body">', unsafe_allow_html=True)

    results          = st.session_state.get("results", {})
    model_choice_res = st.session_state.get("model_choice", model_choice)

    if not results:
        st.markdown("""<div style="text-align:center;padding:4rem 1rem;color:var(--muted)">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:.2">◌</div>
            <div style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase">Awaiting Audio Input</div>
            <div style="font-size:.85rem;margin-top:.8rem;line-height:1.7">Upload a WAV / OGG / MP3 file<br>and click <em>Run Inference</em> to begin.</div>
        </div>""", unsafe_allow_html=True)
    else:
        if model_choice_res == "Both" and "svm" in results and "cnn" in results:
            st.markdown(result_card(results["svm"],"SVM","Support Vector Machine"), unsafe_allow_html=True)
            st.markdown(result_card(results["cnn"],"CNN","Convolutional Neural Network"), unsafe_allow_html=True)
        elif "svm" in results:
            st.markdown(result_card(results["svm"],"SVM","Support Vector Machine"), unsafe_allow_html=True)
        elif "cnn" in results:
            st.markdown(result_card(results["cnn"],"CNN","Convolutional Neural Network"), unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("""<div class="footnote">
    <span>Speech Processing Laboratory · Final Lab Project</span>
    <span class="footnote-accent">PhonemeIQ v2.0</span>
    <span>Group III · FSDD · 2025</span>
</div>""", unsafe_allow_html=True)
