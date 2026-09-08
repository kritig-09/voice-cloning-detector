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
# GLOBAL STYLING
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b0d11;
    color: #e8eaf0;
}

/* Remove excessive top spacing */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* ---------------- HEADER ---------------- */

.hero {
    text-align: center;
    padding: 25px 20px 10px 20px;
}

.hero-icon {
    font-size: 3rem;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f5f7fb;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    max-width: 750px;
    margin: auto;
    line-height: 1.6;
}

/* ---------------- STATUS ---------------- */

.status {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 14px;
    border-radius: 999px;
    background: #111827;
    border: 1px solid #263044;
    color: #a7f3d0;
    font-size: 0.82rem;
    font-weight: 600;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 7px;
}

/* ---------------- METRICS ---------------- */

.metric-box {
    background: #12151c;
    border: 1px solid #242936;
    border-radius: 16px;
    padding: 22px 15px;
    text-align: center;
    min-height: 115px;
}

.metric-number {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f3f4f6;
}

.metric-label {
    margin-top: 5px;
    color: #8f96a5;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ---------------- SECTION ---------------- */

.section-title {
    color: #f1f5f9;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 6px;
}

.section-description {
    color: #8f96a5;
    font-size: 0.9rem;
    margin-bottom: 15px;
}

/* ---------------- BUTTONS ---------------- */

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 11px;
    border: 1px solid #343a4a;
    background: #171b24;
    color: #f3f4f6;
    font-weight: 600;
    font-size: 0.92rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6366f1;
    background: #1b1f2b;
    color: #ffffff;
}

/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {
    background: #12151c;
    border: 1px dashed #3a4050;
    border-radius: 14px;
    padding: 10px;
}

/* ---------------- TABS ---------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #10131a;
    padding: 5px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    padding: 0 20px;
    border-radius: 9px;
    color: #8f96a5;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #1b1f2b !important;
    color: #ffffff !important;
}

/* ---------------- RESULT CARD ---------------- */

.result-real {
    background: #0d2119;
    border: 1px solid #1f6f4d;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.result-ai {
    background: #241416;
    border: 1px solid #7f3037;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.result-title {
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 5px;
}

.result-confidence {
    font-size: 1rem;
    color: #cbd5e1;
}

/* ---------------- INFO CARD ---------------- */

.info-card {
    background: #12151c;
    border: 1px solid #242936;
    border-radius: 14px;
    padding: 18px;
    margin-top: 10px;
}

.info-title {
    color: #f3f4f6;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-text {
    color: #9ca3af;
    font-size: 0.88rem;
    line-height: 1.55;
}

/* ---------------- DIVIDER ---------------- */

hr {
    border: none;
    border-top: 1px solid #20242d;
    margin: 28px 0;
}

/* ---------------- EXPANDER ---------------- */

.streamlit-expanderHeader {
    font-weight: 600;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    color: #626978;
    font-size: 0.78rem;
    padding-top: 25px;
}

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
    model_loaded = False
    model_error = str(e)


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
        AI Voice Cloning Detector
    </div>

    <div class="hero-subtitle">
        Detect AI-generated and cloned voices using machine-learning
        powered audio analysis.
    </div>

    <div class="status">
        <span class="status-dot"></span>
        Detection Engine Online
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP METRICS
# ============================================================

st.write("")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">90%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">5,160</div>
        <div class="metric-label">Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">13</div>
        <div class="metric-label">MFCC Features</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_signal_quality(audio):
    """
    Estimate basic recording quality from RMS level.
    This is not a forensic quality score.
    """

    rms = np.sqrt(np.mean(audio ** 2))

    if rms < 0.01:
        return "Very Low", "Audio level is too low"

    elif rms < 0.03:
        return "Low", "Consider recording a little louder"

    elif rms < 0.08:
        return "Good", "Clear signal level"

    else:
        return "High", "Strong signal level"


def show_spectrogram(audio, sr, title="Audio Spectrogram"):

    fig, ax = plt.subplots(figsize=(9, 3.4))

    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#0b0d11")

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma"
    )

    ax.set_title(
        title,
        fontsize=12,
        color="#e5e7eb",
        pad=12
    )

    ax.set_xlabel("Time", color="#9ca3af")
    ax.set_ylabel("Frequency", color="#9ca3af")

    ax.tick_params(colors="#8f96a5")

    for spine in ax.spines.values():
        spine.set_color("#303542")

    cbar = fig.colorbar(
        ax.collections[0],
        ax=ax,
        pad=0.02
    )

    cbar.ax.tick_params(colors="#8f96a5")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


def show_feature_importance():

    try:

        if not hasattr(model, "feature_importances_"):
            st.info(
                "Feature-level explanation is not available for this model."
            )
            return

        importances = model.feature_importances_

        feature_names = [
            f"MFCC-{i+1}"
            for i in range(len(importances))
        ]

        fig, ax = plt.subplots(figsize=(9, 3.2))

        fig.patch.set_facecolor("#0b0d11")
        ax.set_facecolor("#0b0d11")

        bars = ax.bar(
            feature_names,
            importances
        )

        ax.set_title(
            "MFCC Feature Contribution",
            fontsize=12,
            color="#e5e7eb",
            pad=12
        )

        ax.set_ylabel(
            "Relative Importance",
            color="#9ca3af"
        )

        ax.tick_params(
            axis="x",
            colors="#8f96a5",
            rotation=35
        )

        ax.tick_params(
            axis="y",
            colors="#8f96a5"
        )

        for spine in ax.spines.values():
            spine.set_color("#303542")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.caption(
            "Higher values indicate features that contributed more "
            "to the model's classification."
        )

    except Exception:
        st.info(
            "Feature-level explanation is not available for this model."
        )


def display_result(result_label, confidence):

    confidence_percent = confidence * 100

    if result_label == "Real Voice":

        st.markdown(
            f"""
            <div class="result-real">
                <div class="result-title">
                    ✓ Real Voice
                </div>
                <div class="result-confidence">
                    Confidence: {confidence_percent:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-ai">
                <div class="result-title">
                    ⚠ AI-Cloned Voice Detected
                </div>
                <div class="result-confidence">
                    Confidence: {confidence_percent:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.progress(
        min(max(float(confidence), 0.0), 1.0)
    )


def predict_audio(
    load_path,
    display_name,
    show_analysis=True
):

    if not model_loaded:

        st.error(
            "Model could not be loaded. "
            "Please check voice_cloning_model.pkl."
        )

        return

    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        audio, sr = librosa.load(
            load_path,
            sr=16000,
            mono=True
        )

        duration = len(audio) / sr

        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if duration < 0.5:

            st.warning(
                "Audio is too short. Please provide at least "
                "0.5 seconds of speech."
            )

            return

        peak = np.max(
            np.abs(audio)
        )

        if peak < 0.01:

            st.warning(
                "Audio appears silent or extremely low volume."
            )

            return

        # ----------------------------------------------------
        # SIGNAL QUALITY
        # ----------------------------------------------------

        quality, quality_note = calculate_signal_quality(audio)

        # ----------------------------------------------------
        # FEATURE EXTRACTION
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
        # PREDICTION
        # ----------------------------------------------------

        pred = model.predict(
            mfcc_mean
        )[0]

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                mfcc_mean
            )[0]

            # Existing model convention:
            # class 1 = Real
            # class 0 = AI
            if pred == 1:
                confidence = float(probabilities[1])
            else:
                confidence = float(probabilities[0])

        else:
            confidence = 0.0

        if pred == 1:
            result_label = "Real Voice"
        else:
            result_label = "AI-Cloned Voice"

        # ----------------------------------------------------
        # AUDIO PLAYER
        # ----------------------------------------------------

        if show_analysis:

            st.markdown(
                '<div class="section-title">Audio Analysis</div>',
                unsafe_allow_html=True
            )

            st.audio(
                load_path
            )

            a1, a2, a3 = st.columns(3)

            with a1:
                st.metric(
                    "Duration",
                    f"{duration:.1f} sec"
                )

            with a2:
                st.metric(
                    "Sample Rate",
                    f"{sr / 1000:.0f} kHz"
                )

            with a3:
                st.metric(
                    "Signal Quality",
                    quality
                )

            st.caption(
                f"Signal assessment: {quality_note}"
            )

            st.write("")

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            display_result(
                result_label,
                confidence
            )

            st.write("")

            # ------------------------------------------------
            # SPECTROGRAM
            # ------------------------------------------------

            with st.expander(
                "🎧 View Audio Spectrogram",
                expanded=True
            ):

                show_spectrogram(
                    audio,
                    sr,
                    f"Spectrogram — {display_name}"
                )

            # ------------------------------------------------
            # EXPLAINABILITY
            # ------------------------------------------------

            with st.expander(
                "🔍 Why did the model make this decision?"
            ):

                st.write(
                    "The classifier analyzes 13 Mel-frequency "
                    "cepstral coefficients (MFCCs), which capture "
                    "characteristics of the speech signal."
                )

                show_feature_importance()

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "File": display_name,
                "Result": result_label,
                "Confidence": f"{confidence * 100:.1f}%",
                "Duration": f"{duration:.1f}s"
            }
        )

    except Exception as e:

        st.error(
            f"Unable to process {display_name}."
        )

        st.caption(
            f"Technical details: {str(e)}"
        )


# ============================================================
# MAIN ANALYSIS AREA
# ============================================================

st.markdown(
    '<div class="section-title">Analyze an audio sample</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Test a known sample, record a voice through your microphone,
        or upload an audio file for classification.
    </div>
    """,
    unsafe_allow_html=True
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
# TAB 1 — SAMPLE TEST
# ============================================================

with tab1:

    st.markdown(
        "**Test the detector using prepared audio samples.**"
    )

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "▶ Test Real Human Voice",
            key="real_sample"
        ):

            if os.path.exists("sample_real.flac"):

                predict_audio(
                    "sample_real.flac",
                    "sample_real.flac",
                    True
                )

            else:

                st.error(
                    "sample_real.flac was not found."
                )

    with c2:

        if st.button(
            "▶ Test AI-Cloned Voice",
            key="fake_sample"
        ):

            if os.path.exists("sample_fake.flac"):

                predict_audio(
                    "sample_fake.flac",
                    "sample_fake.flac",
                    True
                )

            else:

                st.error(
                    "sample_fake.flac was not found."
                )


# ============================================================
# TAB 2 — LIVE RECORD
# ============================================================

with tab2:

    st.markdown(
        "**Record a voice directly from your microphone.**"
    )

    st.caption(
        "For best results, speak clearly for 3–10 seconds "
        "in a quiet environment."
    )

    mic_input = st.audio_input(
        "Start a microphone recording",
        key="microphone"
    )

    if mic_input is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(
                mic_input.getbuffer()
            )

            temp_path = temp_file.name

        predict_audio(
            temp_path,
            "Live Microphone Recording",
            True
        )

        try:
            os.unlink(temp_path)
        except Exception:
            pass


# ============================================================
# TAB 3 — UPLOAD
# ============================================================

with tab3:

    st.markdown(
        "**Upload WAV or FLAC audio for analysis.**"
    )

    uploaded_files = st.file_uploader(
        "Choose audio file(s)",
        type=["wav", "flac"],
        accept_multiple_files=True,
        key="audio_upload"
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            predict_audio(
                temp_path,
                uploaded_file.name,
                True
            )

            try:
                os.unlink(temp_path)
            except Exception:
                pass


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Results generated during the current session."
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


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        AI Voice Cloning Detector &nbsp;•&nbsp;
        Machine Learning Audio Analysis
    </div>
    """,
    unsafe_allow_html=True
)
