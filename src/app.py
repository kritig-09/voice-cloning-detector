import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import tempfile


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Voice Cloning Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background-color: #0b0d11;
    color: #e5e7eb;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}


/* ==========================================================
   TYPOGRAPHY
   ========================================================== */

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

h1 {
    font-size: 2.45rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.035em !important;
    color: #f5f7fa !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #f1f3f6 !important;
}

h3 {
    font-size: 1.08rem !important;
    font-weight: 650 !important;
    color: #e7eaf0 !important;
}

p {
    color: #9ba3b2 !important;
}


/* ==========================================================
   SUBTITLE
   ========================================================== */

.app-subtitle {
    color: #8f97a6;
    font-size: 1rem;
    margin-bottom: 0.9rem;
}


/* ==========================================================
   STATUS
   ========================================================== */

.status-text {
    color: #7fcb9a;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 1.5rem;
}


/* ==========================================================
   TECHNICAL STAT STRIP
   NO BOXES
   ========================================================== */

[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.2rem 1.4rem !important;
    min-height: auto !important;
}

[data-testid="stMetricValue"] {
    color: #eef1f5 !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

[data-testid="stMetricLabel"] {
    color: #737c8c !important;
    font-size: 0.72rem !important;
    font-weight: 650 !important;
    letter-spacing: 0.09em !important;
}


/* vertical separators between metrics */

.metrics-divider {
    border-left: 1px solid #292e38;
    height: 48px;
    margin-top: 4px;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color: #252a33 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 1.8rem !important;
}


/* ==========================================================
   TABS
   ========================================================== */

.stTabs [data-baseweb="tab-list"] {
    background-color: #101319;
    border: 1px solid #252a33;
    border-radius: 11px;
    padding: 4px;
    gap: 3px;
}

.stTabs [data-baseweb="tab"] {
    color: #858d9c !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border-radius: 8px;
    padding: 9px 18px;
}

.stTabs [aria-selected="true"] {
    color: #f5f7fa !important;
    background-color: #1b1f27 !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    width: 100%;
    min-height: 45px;
    border-radius: 9px;
    background-color: #151920;
    color: #e9ecf1;
    border: 1px solid #323844;
    font-weight: 600;
    font-size: 0.9rem;
    transition: 0.15s ease;
}

.stButton > button:hover {
    background-color: #1c212a;
    border-color: #606878;
    color: #ffffff;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background-color: #11141a;
    border: 1px dashed #353b47;
    border-radius: 12px;
    padding: 8px;
}


/* ==========================================================
   AUDIO PLAYER
   ========================================================== */

audio {
    width: 100%;
    margin-top: 4px;
    margin-bottom: 8px;
}


/* ==========================================================
   EXPANDERS
   ========================================================== */

[data-testid="stExpander"] {
    background-color: #101319;
    border: 1px solid #272c35;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    font-weight: 600;
}


/* ==========================================================
   ALERTS
   ========================================================== */

[data-testid="stAlert"] {
    border-radius: 10px;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #272c35;
    border-radius: 10px;
    overflow: hidden;
}


/* ==========================================================
   CAPTIONS
   ========================================================== */

[data-testid="stCaptionContainer"] {
    color: #737c8c !important;
}


/* ==========================================================
   PROGRESS BAR
   ========================================================== */

.stProgress > div > div > div > div {
    background-color: #6366f1;
}

</style>
""", unsafe_allow_html=True)


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
        '<div class="status-text">● Detection engine online</div>',
        unsafe_allow_html=True
    )
else:
    st.error("Detection model could not be loaded.")


# ============================================================
# TECHNICAL INFORMATION STRIP
# ============================================================

m1, separator1, m2, separator2, m3 = st.columns(
    [1, 0.05, 1, 0.05, 1]
)

with m1:
    st.metric(
        "MODEL ACCURACY",
        "90%"
    )

with separator1:
    st.markdown(
        '<div class="metrics-divider"></div>',
        unsafe_allow_html=True
    )

with m2:
    st.metric(
        "TRAINING SAMPLES",
        "5,160"
    )

with separator2:
    st.markdown(
        '<div class="metrics-divider"></div>',
        unsafe_allow_html=True
    )

with m3:
    st.metric(
        "MFCC FEATURES",
        "13"
    )


st.divider()


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(audio, sr, filename):

    fig, ax = plt.subplots(
        figsize=(9, 3.4)
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
        color="#dfe3e9",
        fontsize=11,
        pad=10
    )

    ax.set_xlabel(
        "Time",
        color="#8f97a6"
    )

    ax.set_ylabel(
        "Frequency",
        color="#8f97a6"
    )

    ax.tick_params(
        colors="#858d9c"
    )

    for spine in ax.spines.values():
        spine.set_color("#303540")

    cbar = fig.colorbar(
        img,
        ax=ax,
        pad=0.02
    )

    cbar.ax.tick_params(
        colors="#858d9c"
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

    if not hasattr(
        model,
        "feature_importances_"
    ):

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
        figsize=(9, 3)
    )

    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#0b0d11")

    ax.bar(
        feature_names,
        importances
    )

    ax.set_title(
        "MFCC Feature Contribution",
        color="#dfe3e9",
        fontsize=11,
        pad=10
    )

    ax.set_ylabel(
        "Relative Importance",
        color="#8f97a6"
    )

    ax.tick_params(
        axis="x",
        colors="#858d9c",
        rotation=35
    )

    ax.tick_params(
        axis="y",
        colors="#858d9c"
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
# AUDIO PREDICTION
# ============================================================

def predict_audio(
    load_path,
    display_name,
    show_audio=True
):

    if not model_loaded:

        st.error(
            "Detection model is unavailable."
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
        # AUDIO VALIDATION
        # ----------------------------------------------------

        if duration < 0.5:

            st.warning(
                "⚠️ Audio is too short. "
                "Please provide at least 0.5 seconds of speech."
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
        # MFCC FEATURE EXTRACTION
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

        if hasattr(
            model,
            "predict_proba"
        ):

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
                f"✅ Real Voice  •  "
                f"Confidence: {confidence * 100:.1f}%"
            )

        else:

            result_label = "AI-Cloned Voice"

            st.error(
                f"⚠️ AI-Cloned Voice Detected  •  "
                f"Confidence: {confidence * 100:.1f}%"
            )

        st.progress(
            min(
                max(confidence, 0.0),
                1.0
            )
        )

        # ----------------------------------------------------
        # AUDIO DETAILS
        # ----------------------------------------------------

        if show_audio:

            st.subheader("Audio Analysis")

            st.audio(
                load_path
            )

            info1, info2, info3 = st.columns(3)

            with info1:

                st.metric(
                    "Duration",
                    f"{duration:.1f} sec"
                )

            with info2:

                st.metric(
                    "Sample Rate",
                    f"{sr / 1000:.0f} kHz"
                )

            with info3:

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
                    "The detector extracts 13 "
                    "Mel-Frequency Cepstral Coefficients "
                    "(MFCCs) from the audio signal. "
                    "These acoustic features are summarized "
                    "and passed to the trained machine-learning "
                    "classifier for prediction."
                )

                show_feature_importance()

        # ----------------------------------------------------
        # SAVE HISTORY
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
    "Test a known sample, record directly from your microphone, "
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
        "Use the prepared samples to demonstrate the detector."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Test Sample: Real Voice",
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
                    "sample_real.flac not found."
                )

    with col2:

        if st.button(
            "Test Sample: AI-Cloned Voice",
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
                    "sample_fake.flac not found."
                )


# ============================================================
# LIVE MICROPHONE
# ============================================================

with tab2:

    st.write(
        "Record a voice directly using your microphone."
    )

    st.caption(
        "Recommended: 3–10 seconds of clear speech "
        "in a quiet environment."
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
# UPLOAD AUDIO
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

    st.subheader(
        "📜 Prediction History"
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

st.divider()

st.caption(
    "AI Voice Cloning Detector  •  Machine Learning Audio Analysis"
)
