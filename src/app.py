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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background: #0b0d11;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }


    /* ---------- HEADINGS ---------- */

    h1 {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 2.6rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.04em !important;
        color: #f8fafc !important;
    }

    h2 {
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    h3 {
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 650 !important;
        color: #f1f5f9 !important;
    }

    p {
        color: #a8afbd !important;
    }


    /* ---------- TOP TITLE ---------- */

    .app-subtitle {
        color: #8f97a6;
        font-size: 1.05rem;
        margin-top: -12px;
        margin-bottom: 28px;
        letter-spacing: 0.01em;
    }


    /* ---------- STATUS ---------- */

    .status-line {
        display: inline-block;
        padding: 7px 13px;
        border: 1px solid #26352d;
        border-radius: 999px;
        background: #101713;
        color: #8fd5a8;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 22px;
    }


    /* ---------- METRIC CARDS ---------- */

    [data-testid="stMetric"] {
        background: #13161d;
        border: 1px solid #272c36;
        border-radius: 16px;
        padding: 21px 24px;
        min-height: 105px;
    }

    [data-testid="stMetric"]:hover {
        border-color: #3a4050;
    }

    [data-testid="stMetricLabel"] {
        color: #8f97a6 !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
        letter-spacing: 0.08em !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f6fa !important;
        font-size: 1.9rem !important;
        font-weight: 750 !important;
    }


    /* ---------- DIVIDERS ---------- */

    hr {
        border-color: #232832 !important;
        margin-top: 28px !important;
        margin-bottom: 28px !important;
    }


    /* ---------- TABS ---------- */

    .stTabs [data-baseweb="tab-list"] {
        background: #11141a;
        border: 1px solid #242934;
        border-radius: 12px;
        padding: 4px;
        gap: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #9199a8 !important;
        font-weight: 600 !important;
        border-radius: 9px;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: #1b1f28 !important;
        color: #f5f7fb !important;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 10px;
        background: #181c24;
        border: 1px solid #343a47;
        color: #f1f5f9;
        font-size: 0.94rem;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background: #202531;
        border-color: #6366f1;
        color: #ffffff;
    }


    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #11141a;
        border: 1px dashed #3b4250;
        border-radius: 14px;
        padding: 8px;
    }


    /* ---------- AUDIO ---------- */

    audio {
        width: 100%;
        margin-top: 5px;
        margin-bottom: 10px;
    }


    /* ---------- EXPANDERS ---------- */

    [data-testid="stExpander"] {
        background: #11141a;
        border: 1px solid #272c36;
        border-radius: 12px;
    }


    /* ---------- ALERTS ---------- */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #272c36;
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("voice_cloning_model.pkl")
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

st.title("🎙️ AI Voice Cloning Detector")

st.markdown(
    '<div class="app-subtitle">'
    'Real-time detection of AI-generated (cloned) voices vs real human speech'
    '</div>',
    unsafe_allow_html=True
)

if model_loaded:
    st.markdown(
        '<div class="status-line">● Detection engine online</div>',
        unsafe_allow_html=True
    )
else:
    st.error("Detection model could not be loaded.")


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="MODEL ACCURACY",
        value="90%"
    )

with col2:
    st.metric(
        label="TRAINING SAMPLES",
        value="5,160"
    )

with col3:
    st.metric(
        label="MFCC FEATURES USED",
        value="13"
    )


st.divider()


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(audio, sr, filename):

    fig, ax = plt.subplots(
        figsize=(9, 3.5)
    )

    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#0b0d11")

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
        f"Spectrogram — {filename}",
        color="#dce1e8",
        fontsize=12,
        pad=10
    )

    ax.set_xlabel(
        "Time",
        color="#9ca3af"
    )

    ax.set_ylabel(
        "Frequency",
        color="#9ca3af"
    )

    ax.tick_params(
        colors="#9ca3af"
    )

    for spine in ax.spines.values():
        spine.set_color("#303540")

    cbar = fig.colorbar(
        img,
        ax=ax,
        pad=0.02
    )

    cbar.ax.tick_params(
        colors="#9ca3af"
    )

    plt.tight_layout()

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

    fig, ax = plt.subplots(
        figsize=(9, 3.2)
    )

    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#0b0d11")

    ax.bar(
        feature_names,
        importances
    )

    ax.set_title(
        "MFCC Feature Contribution",
        color="#dce1e8",
        fontsize=12,
        pad=10
    )

    ax.set_ylabel(
        "Relative Importance",
        color="#9ca3af"
    )

    ax.tick_params(
        axis="x",
        colors="#9ca3af",
        rotation=35
    )

    ax.tick_params(
        axis="y",
        colors="#9ca3af"
    )

    for spine in ax.spines.values():
        spine.set_color("#303540")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def predict_audio(
    load_path,
    display_name,
    show_audio=True
):

    if not model_loaded:

        st.error(
            "The detection model is unavailable."
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
                "⚠️ Audio is too short. Please provide at least "
                "0.5 seconds of speech."
            )

            return

        peak = np.max(
            np.abs(audio)
        )

        if peak < 0.01:

            st.warning(
                "⚠️ Audio appears silent or has extremely low volume."
            )

            return

        # ----------------------------------------------------
        # SIGNAL QUALITY
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(audio ** 2)
        )

        if rms < 0.01:
            quality = "Very Low"
        elif rms < 0.03:
            quality = "Low"
        elif rms < 0.08:
            quality = "Good"
        else:
            quality = "High"

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
        # PREDICTION
        # ----------------------------------------------------

        pred = model.predict(
            mfcc_mean
        )[0]

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                mfcc_mean
            )[0]

            if pred == 1:

                confidence = float(
                    probabilities[1]
                )

            else:

                confidence = float(
                    probabilities[0]
                )

        else:

            confidence = 0.0

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if pred == 1:

            result_label = "Real Voice"

            st.success(
                f"✅ Real Voice — Confidence: "
                f"{confidence * 100:.1f}%"
            )

        else:

            result_label = "AI-Cloned Voice"

            st.error(
                f"⚠️ AI-Cloned Voice Detected — Confidence: "
                f"{confidence * 100:.1f}%"
            )

        st.progress(
            min(
                max(confidence, 0.0),
                1.0
            )
        )

        # ----------------------------------------------------
        # AUDIO INFORMATION
        # ----------------------------------------------------

        if show_audio:

            st.subheader("Audio Analysis")

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

            st.write("")

            # ------------------------------------------------
            # SPECTROGRAM
            # ------------------------------------------------

            with st.expander(
                "🎧 View Spectrogram",
                expanded=True
            ):

                show_spectrogram(
                    audio,
                    sr,
                    display_name
                )

            # ------------------------------------------------
            # EXPLAINABILITY
            # ------------------------------------------------

            with st.expander(
                "🔍 Why this decision?"
            ):

                st.write(
                    "The model analyzes 13 Mel-Frequency Cepstral "
                    "Coefficients (MFCCs) extracted from the speech "
                    "signal. These features capture characteristics "
                    "of the voice that help the classifier distinguish "
                    "between real and AI-generated speech."
                )

                show_feature_importance()

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "Filename": display_name,
                "Result": result_label,
                "Confidence": f"{confidence * 100:.1f}%",
                "Duration": f"{duration:.1f} sec"
            }
        )

    except Exception as e:

        st.error(
            f"❌ Unable to process {display_name}."
        )

        st.caption(
            f"Technical details: {str(e)}"
        )


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.subheader("Analyze an audio sample")

st.caption(
    "Choose a sample, record directly from your microphone, "
    "or upload an audio file."
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

    st.write(
        "Test the detector using the built-in demonstration samples."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶  Test Sample: Real Voice",
            key="real_sample"
        ):

            if os.path.exists(
                "sample_real.flac"
            ):

                predict_audio(
                    "sample_real.flac",
                    "sample_real.flac"
                )

            else:

                st.error(
                    "sample_real.flac not found in the repository."
                )

    with col2:

        if st.button(
            "▶  Test Sample: AI-Cloned Voice",
            key="fake_sample"
        ):

            if os.path.exists(
                "sample_fake.flac"
            ):

                predict_audio(
                    "sample_fake.flac",
                    "sample_fake.flac"
                )

            else:

                st.error(
                    "sample_fake.flac not found in the repository."
                )


# ============================================================
# LIVE RECORD
# ============================================================

with tab2:

    st.write(
        "Record a voice directly using your microphone."
    )

    st.caption(
        "Recommended: 3–10 seconds of clear speech in a quiet environment."
    )

    mic_input = st.audio_input(
        "Record using your microphone"
    )

    if mic_input is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp:

            temp.write(
                mic_input.getbuffer()
            )

            temp_path = temp.name

        predict_audio(
            temp_path,
            "Microphone Recording"
        )

        try:
            os.remove(temp_path)
        except:
            pass


# ============================================================
# UPLOAD
# ============================================================

with tab3:

    st.write(
        "Upload WAV or FLAC audio for analysis."
    )

    uploaded_files = st.file_uploader(
        "Choose audio file(s)",
        type=["wav", "flac"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp:

                temp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp.name

            predict_audio(
                temp_path,
                uploaded_file.name
            )

            try:
                os.remove(temp_path)
            except:
                pass


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.subheader("📜 Prediction History")

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

st.divider()

st.caption(
    "AI Voice Cloning Detector  •  Machine Learning Audio Analysis"
)
