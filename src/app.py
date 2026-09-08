import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tempfile
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Voice Cloning Detector",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# DARK THEME / CLEAN UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0d11;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main text */
h1, h2, h3 {
    color: #f5f7fb !important;
}

p, label {
    color: #c5cad3 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #12151c;
    border: 1px solid #282d38;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #f5f7fb !important;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: #9299a8 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 46px;
    background-color: #171b24;
    color: #f5f7fb;
    border: 1px solid #343a48;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #6366f1;
    color: white;
    background-color: #1c2030;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background-color: #10131a;
    padding: 5px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #9ca3af;
    font-weight: 600;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    background-color: #1b1f2b !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #12151c;
    border: 1px dashed #3a4050;
    border-radius: 12px;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #11141a;
    border: 1px solid #282d38;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #242832 !important;
}

/* Success / error boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Audio */
audio {
    width: 100%;
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

st.caption(
    "Real-time detection of AI-generated (cloned) voices vs real human speech"
)

if model_loaded:
    st.success("● Detection engine ready for analysis")
else:
    st.error("Model could not be loaded. Check voice_cloning_model.pkl")


st.write("")


# ============================================================
# PROJECT METRICS
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
        label="MFCC FEATURES",
        value="13"
    )


st.divider()


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(audio, sr, filename):

    fig, ax = plt.subplots(figsize=(9, 3.2))

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
        color="#e5e7eb",
        fontsize=12
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
        spine.set_color("#30343e")

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
# MFCC EXPLANATION
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
        color="#e5e7eb",
        fontsize=12
    )

    ax.set_ylabel(
        "Relative Importance",
        color="#9ca3af"
    )

    ax.tick_params(
        colors="#9ca3af",
        axis="x",
        rotation=35
    )

    ax.tick_params(
        colors="#9ca3af",
        axis="y"
    )

    for spine in ax.spines.values():
        spine.set_color("#30343e")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.caption(
        "Higher values indicate features that contributed more "
        "to the model's classification."
    )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def analyze_audio(
    file_path,
    display_name
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
            file_path,
            sr=16000,
            mono=True
        )

        duration = len(audio) / sr

        # ----------------------------------------------------
        # VALIDATION
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
                "Audio appears silent or has extremely low volume."
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

        prediction = model.predict(
            mfcc_mean
        )[0]

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                mfcc_mean
            )[0]

            # Existing model convention:
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

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            result = "Real Voice"

            st.success(
                f"✅ REAL VOICE — Confidence: {confidence * 100:.1f}%"
            )

        else:

            result = "AI-Cloned Voice"

            st.error(
                f"⚠️ AI-CLONED VOICE DETECTED — "
                f"Confidence: {confidence * 100:.1f}%"
            )

        st.progress(
            min(
                max(confidence, 0.0),
                1.0
            )
        )

        st.write("")

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        st.subheader("Audio Analysis")

        st.audio(
            file_path
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

        # ----------------------------------------------------
        # SPECTROGRAM
        # ----------------------------------------------------

        with st.expander(
            "🎧 View Audio Spectrogram",
            expanded=True
        ):

            show_spectrogram(
                audio,
                sr,
                display_name
            )

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        with st.expander(
            "🔍 Why did the model make this decision?"
        ):

            st.write(
                "The detector extracts 13 Mel-Frequency "
                "Cepstral Coefficients (MFCCs) from the audio. "
                "These features represent characteristics of "
                "the speech signal and are passed to the "
                "machine-learning classifier."
            )

            show_feature_importance()

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "File": display_name,
                "Result": result,
                "Confidence": f"{confidence * 100:.1f}%",
                "Duration": f"{duration:.1f} sec"
            }
        )

    except Exception as e:

        st.error(
            "Unable to process this audio file."
        )

        st.caption(
            f"Technical details: {str(e)}"
        )


# ============================================================
# ANALYSIS TABS
# ============================================================

st.subheader("Analyze an audio sample")

st.caption(
    "Test a known sample, record a voice, or upload your own audio."
)


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

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "▶ Test Real Human Voice",
            key="real_voice_button"
        ):

            if os.path.exists(
                "sample_real.flac"
            ):

                analyze_audio(
                    "sample_real.flac",
                    "Real Human Voice"
                )

            else:

                st.error(
                    "sample_real.flac not found."
                )

    with c2:

        if st.button(
            "▶ Test AI-Cloned Voice",
            key="ai_voice_button"
        ):

            if os.path.exists(
                "sample_fake.flac"
            ):

                analyze_audio(
                    "sample_fake.flac",
                    "AI-Cloned Voice"
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
        "For best results, speak clearly for 3–10 seconds "
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

        analyze_audio(
            temp_path,
            "Live Microphone Recording"
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
        "Upload WAV or FLAC audio for detection."
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

            analyze_audio(
                temp_path,
                uploaded_file.name
            )

            try:
                os.remove(temp_path)
            except:
                pass


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📜 Prediction History"
    )

    st.caption(
        "Predictions generated during this session."
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Clear History"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Voice Cloning Detector • Machine Learning Audio Analysis"
)
