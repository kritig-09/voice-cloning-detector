import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import tempfile


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Voice Cloning Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f13; color: #e5e7eb; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1180px; }
.hero { padding: 10px 0 30px 0; }
.hero-icon { font-size: 3rem; margin-bottom: 4px; }
.hero-title { font-size: 3rem; font-weight: 700; letter-spacing: -0.04em; color: #f8fafc; margin-bottom: 8px; }
.hero-subtitle { font-size: 1.05rem; color: #9ca3af; max-width: 760px; line-height: 1.6; }
.status-line { display: flex; align-items: center; gap: 9px; margin-top: 18px; color: #86efac; font-size: 0.88rem; font-weight: 600; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; display: inline-block; box-shadow: 0 0 10px rgba(74,222,128,0.5); }
.stats-row { display: flex; gap: 0; margin-top: 12px; margin-bottom: 30px; }
.stat { flex: 1; padding: 18px 24px; border-left: 1px solid #272b34; }
.stat:first-child { border-left: none; padding-left: 0; }
.stat-label { color: #8f98a8; font-size: 0.76rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 5px; }
.stat-value { color: #e5e7eb; font-size: 1.55rem; font-weight: 700; }
.section-line { border-top: 1px solid #252932; margin: 10px 0 38px 0; }
.section-title { font-size: 1.7rem; font-weight: 650; color: #f1f5f9; margin-bottom: 6px; }
.section-description { color: #858e9d; font-size: 0.95rem; margin-bottom: 20px; }
.info-card { background: #12151b; border: 1px solid #282d37; border-radius: 14px; padding: 22px; height: 100%; }
.info-card-title { font-size: 1rem; font-weight: 600; color: #dce2ea; margin-bottom: 8px; }
.info-card-text { font-size: 0.88rem; color: #858e9d; line-height: 1.6; }
.stButton > button { width: 100%; border-radius: 9px; border: 1px solid #343a46; background: #171a21; color: #dce2ea; font-weight: 600; padding: 0.65rem 1rem; transition: all 0.2s ease; }
.stButton > button:hover { border-color: #6366f1; background: #1b1e27; color: #ffffff; }
[data-testid="stFileUploader"] { background: #12151b; border: 1px dashed #3a404c; border-radius: 12px; padding: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: 5px; border-bottom: 1px solid #252932; }
.stTabs [data-baseweb="tab"] { color: #8992a1; font-weight: 600; padding: 12px 18px; }
.stTabs [aria-selected="true"] { color: #e5e7eb !important; }
.result-real { background: #0d2119; border: 1px solid #174d36; border-radius: 12px; padding: 18px 20px; color: #86efac; font-size: 1.05rem; font-weight: 600; }
.result-ai { background: #251414; border: 1px solid #5a2424; border-radius: 12px; padding: 18px 20px; color: #fca5a5; font-size: 1.05rem; font-weight: 600; }
.audio-metrics { display: flex; gap: 1px; margin: 22px 0; }
.audio-metric { flex: 1; background: #12151b; border: 1px solid #282d37; padding: 18px 20px; }
.audio-metric:first-child { border-radius: 10px 0 0 10px; }
.audio-metric:last-child { border-radius: 0 10px 10px 0; }
.audio-metric-label { color: #8992a1; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.audio-metric-value { color: #e5e7eb; font-size: 1.25rem; font-weight: 700; margin-top: 5px; }
.explain-card { background: #12151b; border: 1px solid #282d37; border-radius: 12px; padding: 22px; margin-top: 20px; }
.explain-title { color: #e5e7eb; font-size: 1rem; font-weight: 600; margin-bottom: 10px; }
.explain-text { color: #929baa; font-size: 0.9rem; line-height: 1.7; }
.footer { border-top: 1px solid #252932; margin-top: 50px; padding-top: 22px; color: #626b79; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "voice_cloning_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
<div class="hero-icon">🎙️</div>
<div class="hero-title">AI Voice Cloning Detector</div>
<div class="hero-subtitle">Real-time detection of AI-generated (cloned) voices vs real human speech using machine-learning based acoustic analysis.</div>
<div class="status-line"><span class="status-dot"></span>Detection engine ready</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP STATS
# ============================================================

st.markdown("""
<div class="stats-row">
<div class="stat"><div class="stat-label">Model Accuracy</div><div class="stat-value">90%</div></div>
<div class="stat"><div class="stat-label">Training Samples</div><div class="stat-value">5,160</div></div>
<div class="stat"><div class="stat-label">MFCC Features</div><div class="stat-value">13</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)


# ============================================================
# SIGNAL QUALITY
# ============================================================

def get_signal_quality(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        return "Very Low"
    elif rms < 0.03:
        return "Low"
    elif rms < 0.08:
        return "Moderate"
    else:
        return "High"


# ============================================================
# WAVEFORM
# ============================================================

def show_waveform(audio, sr, title):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(audio, sr, title):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# MFCC FEATURE IMPORTANCE
# ============================================================

def show_feature_importance():
    if model is None:
        return
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is not available for this classifier.")
        return

    importances = np.asarray(model.feature_importances_)
    feature_names = [f"MFCC-{i + 1}" for i in range(len(importances))]

    indices = np.argsort(importances)
    sorted_importance = importances[indices]
    sorted_names = np.array(feature_names)[indices]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(sorted_names, sorted_importance, color="#6366f1")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("MFCC Feature")
    ax.set_title("MFCC Feature Contribution", fontsize=11)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_audio(load_path, display_name, show_analysis=True):

    if model is None:
        st.error("Model could not be loaded. Make sure voice_cloning_model.pkl is in the repository.")
        return

    try:
        audio, sr = librosa.load(load_path, sr=16000, mono=True)
        duration = len(audio) / sr

        if duration < 0.5:
            st.warning(f"⚠️ {display_name}: Audio is too short.")
            return

        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < 0.01:
            st.warning(f"⚠️ {display_name}: Audio is silent or has very low volume.")
            return

        signal_quality = get_signal_quality(audio)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

        pred = model.predict(mfcc_mean)[0]
        confidence = model.predict_proba(mfcc_mean)[0]

        if pred == 1:
            result_label = "Real Voice"
            confidence_value = float(confidence[1] * 100)
            risk = "Low impersonation risk"
        else:
            result_label = "AI-Cloned Voice"
            confidence_value = float(confidence[0] * 100)
            risk = "High impersonation risk"

        if show_analysis:
            st.markdown(f"### Analysis Result — {display_name}")

            if pred == 1:
                st.markdown(f"""
<div class="result-real">✓ Real Voice &nbsp;•&nbsp; Confidence: {confidence_value:.1f}%<br><span style="font-size:0.82rem;font-weight:400;color:#86a99a;">{risk}</span></div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-ai">⚠ AI-Cloned Voice Detected &nbsp;•&nbsp; Confidence: {confidence_value:.1f}%<br><span style="font-size:0.82rem;font-weight:400;color:#c99b9b;">{risk}</span></div>
""", unsafe_allow_html=True)

            st.progress(min(max(confidence_value / 100, 0.0), 1.0))

            st.markdown("### Audio Analysis")
            st.audio(load_path)

            st.markdown(f"""
<div class="audio-metrics">
<div class="audio-metric"><div class="audio-metric-label">Duration</div><div class="audio-metric-value">{duration:.1f} sec</div></div>
<div class="audio-metric"><div class="audio-metric-label">Sample Rate</div><div class="audio-metric-value">{sr / 1000:.0f} kHz</div></div>
<div class="audio-metric"><div class="audio-metric-label">Signal Quality</div><div class="audio-metric-value">{signal_quality}</div></div>
</div>
""", unsafe_allow_html=True)

            with st.expander("〰️ View Audio Waveform"):
                show_waveform(audio, sr, f"Waveform — {display_name}")

            with st.expander("📊 View Audio Spectrogram"):
                st.caption("Time-frequency representation of the analyzed speech signal.")
                show_spectrogram(audio, sr, f"Spectrogram — {display_name}")

            with st.expander("🔍 Why did the model make this decision?"):
                st.markdown("""
<div class="explain-card">
<div class="explain-title">MFCC-based acoustic analysis</div>
<div class="explain-text">The detector processes the speech signal using <b>Mel-Frequency Cepstral Coefficients (MFCCs)</b>. Thirteen MFCC features are extracted from the audio and summarized before being passed to the trained machine-learning classifier.<br><br>These acoustic features represent characteristics of the speech spectrum and help the classifier distinguish between genuine human speech and AI-generated or cloned speech.</div>
</div>
""", unsafe_allow_html=True)

                st.markdown("#### MFCC Feature Contribution")
                show_feature_importance()
                st.caption("Higher values indicate features that had greater relative importance in the trained classifier.")

        st.session_state.history.append({
            "Filename": display_name,
            "Result": result_label,
            "Confidence": f"{confidence_value:.1f}%",
            "Duration": f"{duration:.1f} sec",
            "Signal Quality": signal_quality
        })

    except Exception as e:
        st.error(f"❌ {display_name}: Error processing audio file.")
        st.caption(f"Technical details: {str(e)}")


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown('<div class="section-title">Analyze an audio sample</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-description">Choose a demonstration sample, record directly from your microphone, or upload your own audio for analysis.</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔊 Sample Test", "🎤 Live Record", "📤 Upload Audio"])


# ============================================================
# SAMPLE TEST
# ============================================================

with tab1:
    st.markdown("#### Test the detector with known samples")
    st.caption("Use these prepared samples to demonstrate the classification pipeline.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="info-card"><div class="info-card-title">Real Human Voice</div><div class="info-card-text">Known genuine speech sample used as a real-voice reference.</div></div>
""", unsafe_allow_html=True)
        st.write("")
        if st.button("▶ Test Sample: Real Voice", key="real_sample"):
            if os.path.exists("sample_real.flac"):
                predict_audio("sample_real.flac", "sample_real.flac", show_analysis=True)
            else:
                st.error("sample_real.flac not found.")

    with col2:
        st.markdown("""
<div class="info-card"><div class="info-card-title">AI-Cloned Voice</div><div class="info-card-text">Known synthetic or cloned speech sample used as an AI-voice reference.</div></div>
""", unsafe_allow_html=True)
        st.write("")
        if st.button("▶ Test Sample: AI-Cloned Voice", key="fake_sample"):
            if os.path.exists("sample_fake.flac"):
                predict_audio("sample_fake.flac", "sample_fake.flac", show_analysis=True)
            else:
                st.error("sample_fake.flac not found.")


# ============================================================
# LIVE RECORD
# ============================================================

with tab2:
    st.markdown("#### Record directly from your microphone")
    st.caption("Record a short speech sample and analyze it using the same detection pipeline.")

    mic_input = st.audio_input("Record using your microphone")

    if mic_input is not None:
        temp_path = os.path.join(tempfile.gettempdir(), "voice_detector_recording.wav")
        with open(temp_path, "wb") as f:
            f.write(mic_input.getbuffer())
        predict_audio(temp_path, "Live Microphone Recording", show_analysis=True)


# ============================================================
# UPLOAD AUDIO
# ============================================================

with tab3:
    st.markdown("#### Upload audio for analysis")
    st.caption("Supported formats: WAV and FLAC")

    uploaded_files = st.file_uploader("Upload audio file(s)", type=["wav", "flac"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(tempfile.gettempdir(), f"voice_detector_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if len(uploaded_files) == 1:
                predict_audio(temp_path, uploaded_file.name, show_analysis=True)
            else:
                predict_audio(temp_path, uploaded_file.name, show_analysis=False)

        if len(uploaded_files) > 1:
            st.success(f"Processed {len(uploaded_files)} audio files.")


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📜 Prediction History</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section-description">Predictions generated during the current session.</div>
""", unsafe_allow_html=True)

    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

    if st.button("Clear Prediction History", key="clear_history"):
        st.session_state.history = []
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">AI Voice Cloning Detector &nbsp;•&nbsp; Machine Learning Audio Analysis &nbsp;•&nbsp; MFCC-based Classification</div>
""", unsafe_allow_html=True)
