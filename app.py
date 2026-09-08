%%writefile app.py
import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({
    'figure.facecolor': '#0f1115',
    'axes.facecolor': '#0f1115',
    'axes.edgecolor': '#3a3d45',
    'axes.labelcolor': '#d1d5db',
    'text.color': '#d1d5db',
    'xtick.color': '#9ca3af',
    'ytick.color': '#9ca3af',
})

st.set_page_config(page_title="Voice Cloning Detector", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0f1115;
        color: #e5e7eb;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Stat cards */
    .metric-card {
        background: #171a21;
        border: 1px solid #262a33;
        border-radius: 14px;
        padding: 22px 10px;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
        margin: 0 0 4px 0;
    }
    .metric-card p {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
        font-weight: 500;
    }

    /* Section headers */
    h3 {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }

    /* Body text */
    p, span, label, .stMarkdown {
        font-size: 0.95rem;
        color: #d1d5db;
    }

    /* Buttons */
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 1.5px dashed #4b5563;
        border-radius: 10px;
        padding: 1em;
        background-color: #171a21;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 10px 18px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
    }

    hr { border-color: #262a33; }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('voice_cloning_model.pkl')

# --- Header ---
st.markdown("<div class='main-title'>🎙️ AI Voice Cloning Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time detection of AI-generated (cloned) voices vs real human speech</div>", unsafe_allow_html=True)

# --- Top stat cards ---
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><h3>90%</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><h3>5,160</h3><p>Training Samples</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><h3>13</h3><p>MFCC Features Used</p></div>', unsafe_allow_html=True)

st.markdown("---")

if "history" not in st.session_state:
    st.session_state.history = []

def show_spectrogram(audio, sr, title):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title, fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)

def show_feature_importance():
    importances = model.feature_importances_
    feature_names = [f"MFCC-{i+1}" for i in range(len(importances))]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(feature_names, importances, color="#6366f1")
    ax.set_title("Which MFCC features influenced this decision most", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    st.pyplot(fig)
    plt.close(fig)

def predict_audio(load_path, display_name, show_audio=True):
    try:
        audio, sr = librosa.load(load_path, sr=16000)
        duration = len(audio) / sr

        if duration < 0.5:
            st.warning(f"⚠️ {display_name}: Audio too short.")
            return
        elif np.max(np.abs(audio)) < 0.01:
            st.warning(f"⚠️ {display_name}: Audio seems silent or very low volume.")
            return

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

        pred = model.predict(mfcc_mean)[0]
        confidence = model.predict_proba(mfcc_mean)[0]

        if show_audio:
            st.audio(load_path)
            st.caption(f"Duration: {duration:.1f} sec")
            show_spectrogram(audio, sr, f"Spectrogram: {display_name}")

        if pred == 1:
            result_label = "Real Voice"
            conf_value = confidence[1]*100
            if show_audio:
                st.success(f"✅ Real Voice — Confidence: {conf_value:.1f}%")
                st.progress(float(confidence[1]))
        else:
            result_label = "AI-Cloned Voice"
            conf_value = confidence[0]*100
            if show_audio:
                st.error(f"⚠️ AI-Cloned Voice Detected — Confidence: {conf_value:.1f}%")
                st.progress(float(confidence[0]))

        if show_audio:
            with st.expander("🔍 Why this decision? (Model Explainability)"):
                show_feature_importance()

        st.session_state.history.append({
            "Filename": display_name,
            "Result": result_label,
            "Confidence": f"{conf_value:.1f}%"
        })

    except Exception as e:
        st.error(f"❌ {display_name}: Error processing audio file. ({str(e)})")

# --- Sections in tabs ---
tab1, tab2, tab3 = st.tabs(["🔊 Sample Test", "🎤 Live Record", "📤 Upload Audio"])

with tab1:
    st.subheader("Try with sample audio")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Sample: Real Voice"):
            predict_audio("sample_real.flac", "sample_real.flac")
    with col2:
        if st.button("Test Sample: AI-Cloned Voice"):
            predict_audio("sample_fake.flac", "sample_fake.flac")

with tab2:
    st.subheader("Record your voice")
    mic_input = st.audio_input("Record using your microphone")
    if mic_input is not None:
        with open("temp_mic.wav", "wb") as f:
            f.write(mic_input.getbuffer())
        predict_audio("temp_mic.wav", "Mic Recording", show_audio=True)

with tab3:
    st.subheader("Upload your own audio")
    uploaded_files = st.file_uploader("Upload audio file(s) (.flac, .wav)",
                                        type=['flac', 'wav'],
                                        accept_multiple_files=True)
    if uploaded_files:
        single_mode = len(uploaded_files) == 1
        for uploaded_file in uploaded_files:
            temp_input = f"temp_{uploaded_file.name}"
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getbuffer())
            load_path = temp_input
            if single_mode:
                predict_audio(load_path, uploaded_file.name, show_audio=True)
            else:
                predict_audio(load_path, uploaded_file.name, show_audio=False)
        if not single_mode:
            st.success(f"Processed {len(uploaded_files)} files — see history below.")

if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    st.table(st.session_state.history)
