import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Voice Cloning Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# MATPLOTLIB THEME
# =========================================================

matplotlib.rcParams.update({
    "figure.facecolor": "#0b0d11",
    "axes.facecolor": "#0b0d11",
    "axes.edgecolor": "#30343d",
    "axes.labelcolor": "#c9ced8",
    "text.color": "#e5e7eb",
    "xtick.color": "#8f96a3",
    "ytick.color": "#8f96a3",
    "font.family": "DejaVu Sans",
})

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background: #0b0d11;
    color: #e5e7eb;
}

.block-container {
    max-width: 1180px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* ---------- FONT ---------- */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ---------- HERO ---------- */

.hero {
    text-align: center;
    padding: 28px 20px 20px 20px;
    margin-bottom: 25px;
}

.hero-icon {
    font-size: 2.5rem;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 3rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #f7f8fa;
    margin: 0;
}

.hero-title span {
    color: #818cf8;
}

.hero-subtitle {
    margin-top: 14px;
    font-size: 1.05rem;
    color: #9ba2af;
    font-weight: 400;
}

.status-line {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 18px;
    padding: 7px 14px;
    border: 1px solid #252a33;
    border-radius: 999px;
    background: #11141a;
    color: #aeb5c2;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: #34d399;
    border-radius: 50%;
    display: inline-block;
}

/* ---------- METRIC CARDS ---------- */

.metric-card {
    background: #11141a;
    border: 1px solid #242933;
    border-radius: 16px;
    padding: 24px 15px;
    text-align: center;
    min-height: 125px;
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #3b4250;
    transform: translateY(-2px);
}

.metric-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f3f4f6;
    margin-bottom: 7px;
}

.metric-label {
    font-size: 0.72rem;
    color: #8f96a3;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

/* ---------- FEATURE CARDS ---------- */

.feature-card {
    background: #0f1217;
    border: 1px solid #20252e;
    border-radius: 12px;
    padding: 15px 18px;
    text-align: center;
    margin-top: 15px;
}

.feature-icon {
    font-size: 1.15rem;
    margin-right: 6px;
}

.feature-title {
    color: #d9dde5;
    font-size: 0.88rem;
    font-weight: 600;
}

.feature-description {
    color: #737b89;
    font-size: 0.72rem;
    margin-top: 4px;
}

/* ---------- SECTION HEADERS ---------- */

.section-title {
    color: #f1f3f6;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 4px;
}

.section-description {
    color: #7f8794;
    font-size: 0.88rem;
    margin-bottom: 18px;
}

/* ---------- TABS ---------- */

.stTabs {
    margin-top: 10px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #0f1217;
    border-bottom: 1px solid #242933;
    padding: 5px;
    border-radius: 10px 10px 0 0;
}

.stTabs [data-baseweb="tab"] {
    color: #858c99;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 11px 18px;
    border-radius: 7px;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #d9dde5;
}

.stTabs [aria-selected="true"] {
    color: #f3f4f6 !important;
    background: #1a1e26 !important;
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    border: 1px solid #303644;
    background: #171b23;
    color: #e7e9ee;
    font-size: 0.9rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6366f1;
    background: #1c2030;
    color: #ffffff;
}

/* ---------- AUDIO ---------- */

audio {
    width: 100%;
    margin-top: 8px;
}

/* ---------- FILE UPLOADER ---------- */

div[data-testid="stFileUploader"] {
    background: #11141a;
    border: 1px dashed #343a46;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;
}

/* ---------- EXPANDER ---------- */

.streamlit-expanderHeader {
    background: #11141a !important;
    border: 1px solid #242933 !important;
    border-radius: 10px !important;
    color: #d9dde5 !important;
    font-weight: 600 !important;
}

.streamlit-expanderContent {
    background: #0f1217 !important;
    border: 1px solid #242933 !important;
    border-top: none !important;
}

/* ---------- RESULT BOX ---------- */

.result-real {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 15px 0;
}

.result-ai {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.25);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 15px 0;
}

.result-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.result-confidence {
    color: #9ca3af;
    font-size: 0.85rem;
}

/* ---------- INFO BOX ---------- */

.info-box {
    background: #101319;
    border: 1px solid #242933;
    border-radius: 12px;
    padding: 15px 18px;
    color: #9299a6;
    font-size: 0.82rem;
    line-height: 1.6;
}

/* ---------- DIVIDER ---------- */

hr {
    border: none;
    border-top: 1px solid #20242c;
    margin: 30px 0;
}

/* ---------- HISTORY ---------- */

.history-title {
    color: #f1f3f6;
    font-size: 1.25rem;
    font-weight: 700;
}

/* ---------- MOBILE ---------- */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1.5rem;
    }

    .hero-title {
        font-size: 2.1rem;
    }

    .hero-subtitle {
        font-size: 0.9rem;
    }

    .metric-card {
        margin-bottom: 10px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("voice_cloning_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the voice detection model.")
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">🎙️</div>

    <div class="hero-title">
        AI Voice <span>Cloning Detector</span>
    </div>

    <div class="hero-subtitle">
        Real-time detection of AI-generated (cloned) voices vs real human speech
    </div>

    <div class="status-line">
        <span class="status-dot"></span>
        MODEL READY FOR ANALYSIS
    </div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL STATS
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">90%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">5,160</div>
        <div class="metric-label">Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">13</div>
        <div class="metric-label">MFCC Features</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PRODUCT FEATURES
# =========================================================

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">
            <span class="feature-icon">🔒</span>
            Privacy-Focused
        </div>
        <div class="feature-description">
            Audio is analyzed for detection without unnecessary processing.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">
            <span class="feature-icon">⚡</span>
            Fast Analysis
        </div>
        <div class="feature-description">
            Get a classification and confidence score within seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">
            <span class="feature-icon">🧠</span>
            ML Powered
        </div>
        <div class="feature-description">
            MFCC-based machine learning analysis for voice classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# SPECTROGRAM
# =========================================================

def show_spectrogram(audio, sr, title):

    fig, ax = plt.subplots(figsize=(8, 3.2))

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma"
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="600",
        pad=10
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def show_feature_importance():

    if not hasattr(model, "feature_importances_"):
        st.info(
            "Feature-level importance is not available for this model."
        )
        return

    importances = model.feature_importances_

    feature_names = [
        f"MFCC-{i+1}"
        for i in range(len(importances))
    ]

    # Sort for cleaner explanation
    indices = np.argsort(importances)

    sorted_importances = importances[indices]
    sorted_names = np.array(feature_names)[indices]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        sorted_names,
        sorted_importances
    )

    ax.set_title(
        "MFCC Features Influencing the Model",
        fontsize=11,
        fontweight="600",
        pad=10
    )

    ax.set_xlabel("Relative Importance")

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_audio(
    load_path,
    display_name,
    show_audio=True,
    add_history=True
):

    try:

        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------

        audio, sr = librosa.load(
            load_path,
            sr=16000
        )

        duration = len(audio) / sr

        # -------------------------------------------------
        # BASIC VALIDATION
        # -------------------------------------------------

        if duration < 0.5:

            st.warning(
                f"⚠️ {display_name}: Audio is too short."
            )

            return

        if len(audio) == 0:

            st.warning(
                f"⚠️ {display_name}: Could not read audio."
            )

            return

        if np.max(np.abs(audio)) < 0.01:

            st.warning(
                f"⚠️ {display_name}: Audio appears silent or too quiet."
            )

            return

        # -------------------------------------------------
        # FEATURE EXTRACTION
        # -------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        ).reshape(1, -1)

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        pred = model.predict(mfcc_mean)[0]

        confidence = model.predict_proba(
            mfcc_mean
        )[0]

        # -------------------------------------------------
        # DETERMINE RESULT
        # -------------------------------------------------

        if pred == 1:

            result_label = "Real Voice"
            conf_value = confidence[1] * 100

        else:

            result_label = "AI-Cloned Voice"
            conf_value = confidence[0] * 100

        # =================================================
        # DISPLAY ANALYSIS
        # =================================================

        if show_audio:

            st.markdown(
                f"""
                <div class="section-title">
                    Analysis Result
                </div>
                """,
                unsafe_allow_html=True
            )

            st.audio(load_path)

            st.caption(
                f"File: {display_name}  •  Duration: {duration:.1f} seconds"
            )

            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            if result_label == "Real Voice":

                st.markdown(
                    f"""
                    <div class="result-real">
                        <div class="result-title">
                            ✅ Real Voice
                        </div>
                        <div class="result-confidence">
                            Model confidence: <strong>{conf_value:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    float(confidence[1])
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-ai">
                        <div class="result-title">
                            ⚠️ AI-Cloned Voice Detected
                        </div>
                        <div class="result-confidence">
                            Model confidence: <strong>{conf_value:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    float(confidence[0])
                )

            # -------------------------------------------------
            # SPECTROGRAM
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Audio Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Time-frequency representation of the analyzed voice signal.'
                '</div>',
                unsafe_allow_html=True
            )

            show_spectrogram(
                audio,
                sr,
                f"Spectrogram — {display_name}"
            )

            # -------------------------------------------------
            # EXPLAINABILITY
            # -------------------------------------------------

            with st.expander(
                "🔍  Why this decision? — Model Explainability"
            ):

                st.markdown(
                    """
                    <div class="info-box">
                        The model uses <strong>13 Mel-Frequency Cepstral
                        Coefficients (MFCCs)</strong> to represent
                        characteristics of the voice signal.
                        <br><br>
                        The chart below shows the relative importance
                        assigned to each MFCC feature by the trained model.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("")

                show_feature_importance()

        # =================================================
        # HISTORY
        # =================================================

        if add_history:

            st.session_state.history.append({
                "Filename": display_name,
                "Result": result_label,
                "Confidence": f"{conf_value:.1f}%"
            })

    except Exception as e:

        st.error(
            f"❌ {display_name}: Error processing audio."
        )

        with st.expander("Technical details"):

            st.code(str(e))


# =========================================================
# MAIN ANALYSIS TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🔊  Sample Test",
    "🎤  Live Record",
    "📤  Upload Audio"
])


# =========================================================
# TAB 1 — SAMPLE TEST
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">Test with sample audio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Use the built-in samples to quickly demonstrate the detector.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶  Test Sample: Real Voice",
            key="real_sample"
        ):

            predict_audio(
                "sample_real.flac",
                "sample_real.flac"
            )

    with col2:

        if st.button(
            "▶  Test Sample: AI-Cloned Voice",
            key="fake_sample"
        ):

            predict_audio(
                "sample_fake.flac",
                "sample_fake.flac"
            )


# =========================================================
# TAB 2 — LIVE RECORDING
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">Record your voice</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Record a short voice sample using your device microphone.'
        '</div>',
        unsafe_allow_html=True
    )

    mic_input = st.audio_input(
        "🎙️ Record using your microphone",
        key="microphone"
    )

    if mic_input is not None:

        temp_mic_path = "temp_mic.wav"

        with open(
            temp_mic_path,
            "wb"
        ) as f:

            f.write(
                mic_input.getbuffer()
            )

        predict_audio(
            temp_mic_path,
            "Live Microphone Recording",
            show_audio=True
        )


# =========================================================
# TAB 3 — UPLOAD
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">Upload audio for analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload WAV or FLAC audio files for AI voice cloning detection.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose audio file(s)",
        type=["flac", "wav"],
        accept_multiple_files=True,
        key="audio_uploader"
    )

    if uploaded_files:

        single_mode = len(uploaded_files) == 1

        for uploaded_file in uploaded_files:

            temp_input = (
                f"temp_{uploaded_file.name}"
            )

            with open(
                temp_input,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            if single_mode:

                predict_audio(
                    temp_input,
                    uploaded_file.name,
                    show_audio=True
                )

            else:

                predict_audio(
                    temp_input,
                    uploaded_file.name,
                    show_audio=False
                )

        if not single_mode:

            st.success(
                f"✓ Successfully processed "
                f"{len(uploaded_files)} audio files."
            )


# =========================================================
# PREDICTION HISTORY
# =========================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown(
        '<div class="history-title">📜 Prediction History</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Results generated during this session."
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Clear Prediction History",
        key="clear_history"
    ):

        st.session_state.history = []

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#646c79;
        font-size:0.75rem;
        padding:10px;
    ">
        AI Voice Cloning Detector • Machine Learning Based Audio Analysis
    </div>
    """,
    unsafe_allow_html=True
)
