import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Voice Cloning Detector",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# MATPLOTLIB STYLE
# ============================================================

matplotlib.rcParams.update({
    "figure.facecolor": "#0d0f14",
    "axes.facecolor": "#0d0f14",
    "axes.edgecolor": "#303642",
    "axes.labelcolor": "#9ca3af",
    "text.color": "#d1d5db",
    "xtick.color": "#8f98a8",
    "ytick.color": "#8f98a8",
})

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ---------------- GLOBAL ---------------- */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d0f14;
    color: #e5e7eb;
}

.main .block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ---------------- HEADER ---------------- */

.hero {
    text-align: center;
    margin-bottom: 2.5rem;
}

.hero-icon {
    font-size: 2.5rem;
    margin-bottom: 0.4rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.1rem;
    line-height: 1.05;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #f8fafc;
    margin: 0;
}

.hero-title span {
    color: #818cf8;
}

.hero-subtitle {
    margin-top: 0.8rem;
    color: #8d96a6;
    font-size: 1rem;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 1rem;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid rgba(74, 222, 128, 0.2);
    background: rgba(74, 222, 128, 0.06);
    color: #86efac;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
}

/* ---------------- SECTION HEADERS ---------------- */

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f3f4f6;
    margin-bottom: 0.25rem;
}

.section-subtitle {
    color: #737c8c;
    font-size: 0.85rem;
    margin-bottom: 1.2rem;
}

/* ---------------- TABS ---------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #222630;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: #7f8898;
    padding: 12px 18px;
}

.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
}

/* ---------------- BUTTONS ---------------- */

.stButton > button {
    width: 100%;
    min-height: 46px;

    background: #181c24;
    color: #e5e7eb;

    border: 1px solid #2c323e;
    border-radius: 11px;

    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;

    transition: all 0.18s ease;
}

.stButton > button:hover {
    background: #202532;
    border-color: #6366f1;
    color: white;
}

/* ---------------- FILE UPLOADER ---------------- */

div[data-testid="stFileUploader"] section {
    background: #12151b !important;
    border: 1.5px dashed #353b47 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #6366f1 !important;
}

/* ---------------- AUDIO ---------------- */

.stAudio {
    margin-top: 0.5rem;
}

/* ---------------- RESULT ---------------- */

.result-card {
    margin: 1.5rem 0;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
}

.result-real {
    background: linear-gradient(
        135deg,
        rgba(34, 197, 94, 0.09),
        rgba(17, 24, 39, 0.9)
    );
    border: 1px solid rgba(74, 222, 128, 0.22);
}

.result-ai {
    background: linear-gradient(
        135deg,
        rgba(239, 68, 68, 0.09),
        rgba(17, 24, 39, 0.9)
    );
    border: 1px solid rgba(248, 113, 113, 0.22);
}

.result-small {
    color: #7f8999;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-top: 7px;
}

.real-title {
    color: #86efac;
}

.ai-title {
    color: #fca5a5;
}

.confidence {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.7rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 8px;
}

.confidence-label {
    color: #747e8e;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ---------------- INFO CHIPS ---------------- */

.info-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 1rem 0;
}

.info-chip {
    background: #151922;
    border: 1px solid #282e39;
    border-radius: 8px;
    padding: 6px 10px;
    color: #929baa;
    font-size: 0.73rem;
}

/* ---------------- FORENSICS ---------------- */

.forensics-header {
    font-family: 'Space Grotesk', sans-serif;
    color: #f1f5f9;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 1.8rem;
}

.forensics-description {
    color: #6f7888;
    font-size: 0.78rem;
    margin-bottom: 0.8rem;
}

/* ---------------- EXPLAINABILITY ---------------- */

.explain-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #f1f5f9;
    font-size: 1.05rem;
    font-weight: 600;
}

.explain-description {
    color: #6f7888;
    font-size: 0.75rem;
    margin-bottom: 0.8rem;
}

/* ---------------- TECHNICAL ---------------- */

.tech-box {
    background: #11141a;
    border: 1px solid #242a34;
    border-radius: 12px;
    padding: 15px 18px;
}

.tech-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #20242c;
    font-size: 0.8rem;
}

.tech-row:last-child {
    border-bottom: none;
}

.tech-key {
    color: #737d8c;
}

.tech-value {
    color: #cbd5e1;
}

/* ---------------- DIVIDER ---------------- */

hr {
    border-color: #222630 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("voice_cloning_model.pkl")


model = load_model()


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

    <div class="hero-title">
        AI Voice <span>Cloning Detector</span>
    </div>

    <div class="hero-subtitle">
        Detect AI-generated and cloned voices using machine-learning
        powered audio analysis.
    </div>

    <div class="status">
        <span class="status-dot"></span>
        System Ready
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SPECTROGRAM
# ============================================================

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
        fontweight="500",
        pad=10
    )

    ax.set_xlabel(
        "Time",
        fontsize=9
    )

    ax.set_ylabel(
        "Frequency",
        fontsize=9
    )

    ax.tick_params(
        labelsize=8
    )

    cbar = fig.colorbar(
        img,
        ax=ax,
        pad=0.02
    )

    cbar.set_label(
        "dB",
        fontsize=8
    )

    cbar.ax.tick_params(
        labelsize=8
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# MFCC FEATURE IMPORTANCE
# ============================================================

def show_feature_importance():

    if not hasattr(model, "feature_importances_"):
        st.info(
            "Feature importance is not available for this model."
        )
        return

    importances = model.feature_importances_

    feature_names = [
        f"MFCC-{i + 1}"
        for i in range(len(importances))
    ]

    # Sort features by importance
    order = np.argsort(importances)

    fig, ax = plt.subplots(
        figsize=(8, 3)
    )

    ax.barh(
        np.array(feature_names)[order],
        importances[order],
        height=0.55
    )

    ax.set_xlabel(
        "Relative importance",
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.grid(
        axis="x",
        alpha=0.08
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# RESULT DISPLAY
# ============================================================

def show_result(result_label, conf_value):

    if result_label == "Real Voice":

        st.markdown(
            f"""
            <div class="result-card result-real">

                <div class="result-small">
                    Analysis Complete
                </div>

                <div class="result-title real-title">
                    ✓ Human Voice
                </div>

                <div class="confidence">
                    {conf_value:.1f}%
                </div>

                <div class="confidence-label">
                    Detection Confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            float(conf_value / 100)
        )

    else:

        st.markdown(
            f"""
            <div class="result-card result-ai">

                <div class="result-small">
                    Analysis Complete
                </div>

                <div class="result-title ai-title">
                    ⚠ AI-Cloned Voice Detected
                </div>

                <div class="confidence">
                    {conf_value:.1f}%
                </div>

                <div class="confidence-label">
                    Detection Confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            float(conf_value / 100)
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_audio(
    load_path,
    display_name,
    show_audio=True
):

    try:

        # Load audio
        audio, sr = librosa.load(
            load_path,
            sr=16000
        )

        duration = len(audio) / sr

        # Validation
        if duration < 0.5:

            st.warning(
                f"⚠️ {display_name}: Audio is too short."
            )

            return

        if np.max(np.abs(audio)) < 0.01:

            st.warning(
                f"⚠️ {display_name}: Audio seems silent or very low volume."
            )

            return

        # MFCC extraction
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        ).reshape(1, -1)

        # Prediction
        pred = model.predict(
            mfcc_mean
        )[0]

        confidence = model.predict_proba(
            mfcc_mean
        )[0]

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if pred == 1:

            result_label = "Real Voice"
            conf_value = confidence[1] * 100

        else:

            result_label = "AI-Cloned Voice"
            conf_value = confidence[0] * 100

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if show_audio:

            show_result(
                result_label,
                conf_value
            )

            # Audio information
            st.markdown(
                f"""
                <div class="info-row">

                    <span class="info-chip">
                        🎵 {display_name}
                    </span>

                    <span class="info-chip">
                        ⏱ {duration:.1f} sec
                    </span>

                    <span class="info-chip">
                        16 kHz
                    </span>

                    <span class="info-chip">
                        Mono
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            # Audio player
            st.audio(
                load_path
            )

            # ------------------------------------------------
            # AUDIO FORENSICS
            # ------------------------------------------------

            st.markdown(
                """
                <div class="forensics-header">
                    Audio Forensics
                </div>

                <div class="forensics-description">
                    Visual representation of the analyzed audio signal.
                </div>
                """,
                unsafe_allow_html=True
            )

            show_spectrogram(
                audio,
                sr,
                f"Spectrogram · {display_name}"
            )

            # ------------------------------------------------
            # EXPLAINABILITY
            # ------------------------------------------------

            with st.expander(
                "🔍  Model Explainability"
            ):

                st.markdown(
                    """
                    <div class="explain-title">
                        MFCC Feature Importance
                    </div>

                    <div class="explain-description">
                        Relative importance of the extracted MFCC features
                        used by the trained classification model.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                show_feature_importance()

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.session_state.history.append({
            "Audio": display_name,
            "Result": result_label,
            "Confidence": f"{conf_value:.1f}%"
        })

    except Exception as e:

        st.error(
            f"❌ {display_name}: Error processing audio file. "
            f"({str(e)})"
        )


# ============================================================
# MAIN SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Analyze an audio sample
    </div>

    <div class="section-subtitle">
        Choose a sample, record live audio, or upload your own recording.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔊 Sample Test",
    "🎤 Live Record",
    "📤 Upload Audio"
])


# ============================================================
# SAMPLE TEST
# ============================================================

with tab1:

    st.markdown(
        "#### Try the detector with a sample"
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        if st.button(
            "✓  Test Real Voice",
            key="real_voice"
        ):

            predict_audio(
                "sample_real.flac",
                "sample_real.flac"
            )

    with col2:

        if st.button(
            "⚠  Test AI-Cloned Voice",
            key="ai_voice"
        ):

            predict_audio(
                "sample_fake.flac",
                "sample_fake.flac"
            )


# ============================================================
# LIVE RECORD
# ============================================================

with tab2:

    st.markdown(
        "#### Record your voice"
    )

    st.caption(
        "Record a short speech sample using your microphone."
    )

    mic_input = st.audio_input(
        "Record using your microphone"
    )

    if mic_input is not None:

        with open(
            "temp_mic.wav",
            "wb"
        ) as f:

            f.write(
                mic_input.getbuffer()
            )

        predict_audio(
            "temp_mic.wav",
            "Microphone Recording",
            show_audio=True
        )


# ============================================================
# UPLOAD AUDIO
# ============================================================

with tab3:

    st.markdown(
        "#### Upload your own audio"
    )

    st.caption(
        "Supported formats: WAV and FLAC"
    )

    uploaded_files = st.file_uploader(
        "Drop your audio file here",
        type=["flac", "wav"],
        accept_multiple_files=True
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

            predict_audio(
                temp_input,
                uploaded_file.name,
                show_audio=single_mode
            )

        if not single_mode:

            st.success(
                f"Processed {len(uploaded_files)} audio files."
            )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.markdown("---")

with st.expander(
    "⚙️  Technical Details"
):

    st.markdown(
        """
        <div class="tech-box">

            <div class="tech-row">
                <span class="tech-key">Model accuracy</span>
                <span class="tech-value">90%</span>
            </div>

            <div class="tech-row">
                <span class="tech-key">Training samples</span>
                <span class="tech-value">5,160</span>
            </div>

            <div class="tech-row">
                <span class="tech-key">Audio sample rate</span>
                <span class="tech-value">16 kHz</span>
            </div>

            <div class="tech-row">
                <span class="tech-key">Features</span>
                <span class="tech-value">13 MFCC coefficients</span>
            </div>

            <div class="tech-row">
                <span class="tech-key">Supported formats</span>
                <span class="tech-value">WAV / FLAC</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown(
        """
        <div class="section-title">
            Prediction History
        </div>

        <div class="section-subtitle">
            Audio analyses completed during this session.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )
