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

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');


/* ==========================================================
   GLOBAL
   ========================================================== */

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #0f1115;
    color: #e5e7eb;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   TYPOGRAPHY
   ========================================================== */

h1 {
    font-family: 'Inter', sans-serif !important;
    color: #f8fafc !important;
    font-size: 2.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    color: #f1f3f6 !important;
    font-weight: 650 !important;
}

h3 {
    color: #eef1f5 !important;
    font-weight: 600 !important;
}

p {
    color: #a3aab7 !important;
}


/* ==========================================================
   HEADER
   ========================================================== */

.header-subtitle {
    color: #9299a7;
    font-size: 0.98rem;
    font-weight: 400;
    margin-bottom: 0.8rem;
}

.engine-status {
    color: #7fd29b;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.025em;
    margin-bottom: 1.8rem;
}


/* ==========================================================
   TECHNICAL INFO STRIP
   ========================================================== */

[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-height: auto !important;
}

[data-testid="stMetricLabel"] {
    color: #737b8b !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetricValue"] {
    color: #e8ebf0 !important;
    font-size: 1.25rem !important;
    font-weight: 650 !important;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color: #282d36 !important;
    margin-top: 1.7rem !important;
    margin-bottom: 1.7rem !important;
}


/* ==========================================================
   TABS
   ========================================================== */

.stTabs [data-baseweb="tab-list"] {
    background-color: #15181f;
    border: 1px solid #292e38;
    border-radius: 11px;
    padding: 4px;
    gap: 3px;
}

.stTabs [data-baseweb="tab"] {
    color: #8f97a6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border-radius: 8px;
    padding: 9px 18px;
}

.stTabs [aria-selected="true"] {
    color: #f5f7fa !important;
    background-color: #20242d !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    width: 100%;
    min-height: 45px;
    background-color: #191c23;
    color: #e8ebef;
    border: 1px solid #343a46;
    border-radius: 9px;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem;
    font-weight: 600;
    transition: 0.15s ease;
}

.stButton > button:hover {
    background-color: #20242c;
    border-color: #596171;
    color: #ffffff;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background-color: #15181f;
    border: 1px dashed #3b414d;
    border-radius: 12px;
    padding: 8px;
}


/* ==========================================================
   AUDIO PLAYER
   ========================================================== */

audio {
    width: 100%;
}


/* ==========================================================
   EXPANDERS
   ========================================================== */

[data-testid="stExpander"] {
    background-color: #13161c;
    border: 1px solid #292e37;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    color: #e7eaf0 !important;
    font-weight: 600;
}


/* ==========================================================
   ALERTS
   ========================================================== */

[data-testid="stAlert"] {
    border-radius: 10px;
}


/* ==========================================================
   PROGRESS
   ========================================================== */

.stProgress > div > div > div > div {
    background-color: #6366f1;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #292e37;
    border-radius: 10px;
    overflow: hidden;
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
    '<div class="header-subtitle">'
    'Real-time detection of AI-generated (cloned) voices vs real human speech'
    '</div>',
    unsafe_allow_html=True
)

if model_loaded:
    st.markdown(
        '<div class="engine-status">'
        '● Detection engine ready'
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.error(
        "Detection model could not be loaded."
    )


# ============================================================
# MODEL INFORMATION
# Small technical information instead of large cards
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "MODEL ACCURACY",
        "90%"
    )

with info2:
    st.metric(
        "TRAINING SAMPLES",
        "5,160"
    )

with info3:
    st.metric(
        "MFCC FEATURES",
        "13"
    )


st.divider()


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(
    audio,
    sr,
    filename
):

    fig, ax = plt.subplots(
        figsize=(9, 3.2)
    )

    fig.patch.set_facecolor("#0f1115")
    ax.set_facecolor("#0f1115")

    D = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(audio)
        ),
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
        fontsize=11,
        color="#dfe3e8",
        pad=9
    )

    ax.set_xlabel(
        "Time",
        color="#8e96a5"
    )

    ax.set_ylabel(
        "Frequency",
        color="#8e96a5"
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
            "Feature importance is not available "
            "for this classifier."
        )

        return

    importances = model.feature_importances_

    feature_names = [
        f"MFCC-{i + 1}"
        for i in range(
            len(importances)
        )
    ]

    fig, ax = plt.subplots(
        figsize=(9, 3.1)
    )

    fig.patch.set_facecolor("#0f1115")
    ax.set_facecolor("#0f1115")

    ax.bar(
        feature_names,
        importances
    )

    ax.set_title(
        "MFCC Feature Contribution",
        color="#dfe3e8",
        fontsize=11,
        pad=9
    )

    ax.set_ylabel(
        "Relative Importance",
        color="#8e96a5"
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

    st.caption(
        "Higher values indicate features that contributed "
        "more strongly to the classifier's decision."
    )


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
                "Audio is too short. "
                "Please provide at least 0.5 seconds of speech."
            )

            return

        peak = np.max(
            np.abs(audio)
        )

        if peak < 0.01:

            st.warning(
                "Audio appears silent or has very low volume."
            )

            return


        # ----------------------------------------------------
        # SIGNAL QUALITY
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(
                audio ** 2
            )
        )

        if rms < 0.01:
            signal_quality = "Very Low"

        elif rms < 0.03:
            signal_quality = "Low"

        elif rms < 0.08:
            signal_quality = "Good"

        else:
            signal_quality = "High"


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
        ).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            mfcc_mean
        )[0]


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                mfcc_mean
            )[0]

            # Existing model mapping:
            # 1 = Real Voice
            # 0 = AI-Cloned Voice

            if prediction == 1:

                confidence = float(
                    probabilities[1]
                )

            else:

                confidence = float(
                    probabilities[0]
                )

        else:

            confidence = 0.0


        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

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
                max(
                    confidence,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # AUDIO ANALYSIS
        # ====================================================

        if show_audio:

            st.subheader(
                "Audio Analysis"
            )

            st.audio(
                load_path
            )


            # ------------------------------------------------
            # THREE PARAMETERS
            # ------------------------------------------------

            param1, param2, param3 = st.columns(3)

            with param1:

                st.metric(
                    "DURATION",
                    f"{duration:.1f} sec"
                )

            with param2:

                st.metric(
                    "SAMPLE RATE",
                    f"{sr / 1000:.0f} kHz"
                )

            with param3:

                st.metric(
                    "SIGNAL QUALITY",
                    signal_quality
                )


            st.write("")


            # ------------------------------------------------
            # SPECTROGRAM
            # ------------------------------------------------

            with st.expander(
                "🎧 Audio Spectrogram",
                expanded=True
            ):

                st.caption(
                    "Time-frequency representation of the analyzed speech signal."
                )

                show_spectrogram(
                    audio,
                    sr,
                    display_name
                )


            # ------------------------------------------------
            # MODEL EXPLAINABILITY
            # ------------------------------------------------

            with st.expander(
                "🔍 Why did the model make this decision?"
            ):

                st.write(
                    "The detector processes the speech signal "
                    "using Mel-Frequency Cepstral Coefficients "
                    "(MFCCs). Thirteen MFCC features are extracted "
                    "from the audio and summarized before being "
                    "passed to the trained classifier."
                )

                st.write(
                    "These acoustic features capture characteristics "
                    "of the speech signal that help the model "
                    "differentiate between real human speech and "
                    "AI-generated or cloned speech."
                )

                show_feature_importance()


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        st.session_state.history.append(
            {
                "Filename": display_name,
                "Result": result_label,
                "Confidence": f"{confidence * 100:.1f}%",
                "Duration": f"{duration:.1f} sec",
                "Signal Quality": signal_quality
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
# ANALYSIS SECTION
# ============================================================

st.subheader(
    "Analyze an audio sample"
)

st.caption(
    "Choose a demonstration sample, record directly from your "
    "microphone, or upload your own audio."
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

    st.write("")


    sample_col1, sample_col2 = st.columns(
        2
    )


    # --------------------------------------------------------
    # REAL VOICE
    # --------------------------------------------------------

    with sample_col1:

        st.markdown(
            "**Real Human Voice**"
        )

        st.caption(
            "Known genuine speech sample"
        )

        if st.button(
            "▶  Test Sample: Real Voice",
            key="real_voice"
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


    # --------------------------------------------------------
    # AI VOICE
    # --------------------------------------------------------

    with sample_col2:

        st.markdown(
            "**AI-Cloned Voice**"
        )

        st.caption(
            "Known synthetic / cloned speech sample"
        )

        if st.button(
            "▶  Test Sample: AI-Cloned Voice",
            key="fake_voice"
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
# LIVE RECORD
# ============================================================

with tab2:

    st.write(
        "Record a voice directly using your microphone."
    )

    st.caption(
        "Recommended recording length: 3–10 seconds. "
        "For best results, speak clearly in a quiet environment."
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

            os.remove(
                temp_path
            )

        except Exception:

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
        type=[
            "wav",
            "flac"
        ],
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

                os.remove(
                    temp_path
                )

            except Exception:

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
        "Predictions generated during the current session."
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
    "AI Voice Cloning Detector  •  "
    "Machine Learning Audio Analysis"
)
