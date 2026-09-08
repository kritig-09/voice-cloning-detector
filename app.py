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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MATPLOTLIB DARK THEME
# ============================================================

matplotlib.rcParams.update({
    "figure.facecolor": "#0d0f14",
    "axes.facecolor": "#0d0f14",
    "axes.edgecolor": "#303641",
    "axes.labelcolor": "#a1a8b5",
    "text.color": "#d7dbe3",
    "xtick.color": "#8c95a5",
    "ytick.color": "#8c95a5",
    "font.family": "DejaVu Sans",
})


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');


/* ==========================================================
   GLOBAL
   ========================================================== */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d0f14;
    color: #e5e7eb;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ==========================================================
   HEADER
   ========================================================== */

h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    color: #f8fafc !important;
}

.header-subtitle {
    text-align: center;
    color: #8f98a8;
    font-size: 0.98rem;
    margin-top: -0.8rem;
    margin-bottom: 1.4rem;
}

.system-status {
    text-align: center;
    color: #86efac;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 2.3rem;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f1f5f9 !important;
}

h3 {
    font-size: 1.15rem !important;
}


/* ==========================================================
   TABS
   ========================================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #242933;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    color: #8b94a4;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 12px 18px;
}

.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    width: 100%;
    min-height: 46px;

    background: #171b23;
    color: #e5e7eb;

    border: 1px solid #303641;
    border-radius: 10px;

    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;

    transition: all 0.18s ease;
}

.stButton > button:hover {
    background: #202532;
    border-color: #6366f1;
    color: #ffffff;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

div[data-testid="stFileUploader"] section {
    background: #11141a !important;
    border: 1.5px dashed #353c49 !important;
    border-radius: 13px !important;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #6366f1 !important;
}


/* ==========================================================
   AUDIO PLAYER
   ========================================================== */

audio {
    border-radius: 10px;
}


/* ==========================================================
   RESULT BOXES
   ========================================================== */

.real-result {
    background: #101b16;
    border: 1px solid #214d37;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 1.3rem 0;
}

.ai-result {
    background: #1b1215;
    border: 1px solid #5b2b32;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 1.3rem 0;
}

.result-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7f8998;
}

.result-real-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #86efac;
    margin-top: 5px;
}

.result-ai-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fca5a5;
    margin-top: 5px;
}

.result-confidence {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 6px;
}

.confidence-label {
    color: #737d8d;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}


/* ==========================================================
   AUDIO INFO
   ========================================================== */

.audio-info {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 0.9rem 0 1.2rem 0;
}

.audio-chip {
    background: #151922;
    border: 1px solid #282e39;
    border-radius: 7px;
    padding: 5px 10px;
    color: #929baa;
    font-size: 0.72rem;
}


/* ==========================================================
   FORENSICS SECTION
   ========================================================== */

.forensics-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-top: 1.7rem;
}

.forensics-subtitle {
    color: #727c8c;
    font-size: 0.78rem;
    margin-bottom: 0.8rem;
}


/* ==========================================================
   EXPANDERS
   ========================================================== */

.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}


/* ==========================================================
   PROGRESS BAR
   ========================================================== */

.stProgress > div > div > div > div {
    background-color: #6366f1;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #252b35;
    border-radius: 10px;
}


/* ==========================================================
   DIVIDER
   ========================================================== */

hr {
    border-color: #242933 !important;
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
}


/* ==========================================================
   CAPTION
   ========================================================== */

.stCaption {
    color: #727c8c !important;
}


/* ==========================================================
   TECHNICAL DETAILS
   ========================================================== */

.tech-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 2px;
    border-bottom: 1px solid #20252e;
    font-size: 0.8rem;
}

.tech-row:last-child {
    border-bottom: none;
}

.tech-name {
    color: #727c8c;
}

.tech-value {
    color: #cbd5e1;
    font-weight: 500;
}

</style>
""",
    unsafe_allow_html=True
)


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

st.markdown(
    """
    <h1 style="
        text-align:center;
        font-size:3rem;
        margin-bottom:0.4rem;
    ">
        🎙️ AI Voice Cloning Detector
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-subtitle">
        Detect AI-generated and cloned voices using machine-learning
        powered audio analysis.
    </div>

    <div class="system-status">
        ● SYSTEM READY
    </div>
    """,
    unsafe_allow_html=True
)


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
# FEATURE IMPORTANCE
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
            <div class="real-result">

                <div class="result-label">
                    Analysis Complete
                </div>

                <div class="result-real-title">
                    ✓ Human Voice
                </div>

                <div class="result-confidence">
                    {conf_value:.1f}%
                </div>

                <div class="confidence-label">
                    Detection Confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="ai-result">

                <div class="result-label">
                    Analysis Complete
                </div>

                <div class="result-ai-title">
                    ⚠ AI-Cloned Voice Detected
                </div>

                <div class="result-confidence">
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
# PREDICT AUDIO
# ============================================================

def predict_audio(
    load_path,
    display_name,
    show_audio=True
):

    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        audio, sr = librosa.load(
            load_path,
            sr=16000
        )

        duration = len(audio) / sr

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if duration < 0.5:

            st.warning(
                f"⚠️ {display_name}: Audio is too short."
            )

            return

        if np.max(np.abs(audio)) < 0.01:

            st.warning(
                f"⚠️ {display_name}: Audio seems silent or "
                f"very low volume."
            )

            return

        # ----------------------------------------------------
        # MFCC EXTRACTION
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        ).reshape(1, -1)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        pred = model.predict(
            mfcc_mean
        )[0]

        confidence = model.predict_proba(
            mfcc_mean
        )[0]

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if pred == 1:

            result_label = "Real Voice"
            conf_value = confidence[1] * 100

        else:

            result_label = "AI-Cloned Voice"
            conf_value = confidence[0] * 100

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if show_audio:

            show_result(
                result_label,
                conf_value
            )

            # ------------------------------------------------
            # AUDIO INFORMATION
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="audio-info">

                    <span class="audio-chip">
                        🎵 {display_name}
                    </span>

                    <span class="audio-chip">
                        ⏱ {duration:.1f} sec
                    </span>

                    <span class="audio-chip">
                        16 kHz
                    </span>

                    <span class="audio-chip">
                        13 MFCC
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # AUDIO PLAYER
            # ------------------------------------------------

            st.audio(
                load_path
            )

            # ------------------------------------------------
            # FORENSICS
            # ------------------------------------------------

            st.markdown(
                """
                <div class="forensics-title">
                    Audio Forensics
                </div>

                <div class="forensics-subtitle">
                    Spectral representation of the analyzed audio signal.
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
                    The model uses 13 Mel-Frequency Cepstral
                    Coefficients (MFCCs) to characterize the
                    audio signal.

                    The chart below shows the relative importance
                    assigned to each feature by the classifier.
                    """
                )

                show_feature_importance()

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "Audio": display_name,
                "Result": result_label,
                "Confidence": f"{conf_value:.1f}%"
            }
        )

    except Exception as e:

        st.error(
            f"❌ {display_name}: Error processing audio file. "
            f"({str(e)})"
        )


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown(
    """
    <h2 style="
        font-size:1.35rem;
        margin-bottom:0.2rem;
    ">
        Analyze an audio sample
    </h2>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Choose a sample, record live audio, or upload your own recording."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔊 Sample Test",
        "🎤 Live Record",
        "📤 Upload Audio"
    ]
)


# ============================================================
# SAMPLE TEST
# ============================================================

with tab1:

    st.markdown(
        "#### Test the detector"
    )

    st.caption(
        "Use the provided recordings to quickly verify the system."
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
# UPLOAD
# ============================================================

with tab3:

    st.markdown(
        "#### Upload your own audio"
    )

    st.caption(
        "Supported formats: WAV and FLAC."
    )

    uploaded_files = st.file_uploader(
        "Drop your audio file here",
        type=[
            "flac",
            "wav"
        ],
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
        <div class="tech-row">
            <span class="tech-name">Model accuracy</span>
            <span class="tech-value">90%</span>
        </div>

        <div class="tech-row">
            <span class="tech-name">Training samples</span>
            <span class="tech-value">5,160</span>
        </div>

        <div class="tech-row">
            <span class="tech-name">Audio sample rate</span>
            <span class="tech-value">16 kHz</span>
        </div>

        <div class="tech-row">
            <span class="tech-name">Feature extraction</span>
            <span class="tech-value">13 MFCC coefficients</span>
        </div>

        <div class="tech-row">
            <span class="tech-name">Supported formats</span>
            <span class="tech-value">WAV / FLAC</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown(
        """
        <h2 style="
            font-size:1.25rem;
            margin-bottom:0.15rem;
        ">
            Prediction History
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Audio analyses completed during this session."
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )
