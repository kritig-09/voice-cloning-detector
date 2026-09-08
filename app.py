import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os
import hashlib

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
# MATPLOTLIB THEME
# ============================================================

matplotlib.rcParams.update({
    "figure.facecolor": "#0b0d11",
    "axes.facecolor": "#0b0d11",
    "axes.edgecolor": "#292e38",
    "axes.labelcolor": "#9ca3af",
    "text.color": "#e5e7eb",
    "xtick.color": "#8b93a3",
    "ytick.color": "#8b93a3",
    "font.family": "DejaVu Sans",
})

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* =========================================================
   GLOBAL
   ========================================================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 50% -15%,
            rgba(99, 102, 241, 0.10),
            transparent 38%
        ),
        #0b0d11;
    color: #e5e7eb;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 3.2rem;
    padding-bottom: 4rem;
}

/* Hide Streamlit decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    text-align: center;
    padding: 10px 0 34px 0;
}

.hero-icon {
    width: 64px;
    height: 64px;
    margin: 0 auto 18px auto;
    border-radius: 18px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        135deg,
        rgba(99, 102, 241, 0.20),
        rgba(139, 92, 246, 0.10)
    );

    border: 1px solid rgba(129, 140, 248, 0.25);

    font-size: 30px;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.6rem, 5vw, 4.2rem);
    line-height: 0.98;
    letter-spacing: -0.055em;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
}

.hero h1 .accent {
    color: #818cf8;
}

.hero-subtitle {
    max-width: 650px;
    margin: 18px auto 0 auto;

    font-size: 1rem;
    line-height: 1.65;

    color: #8f98a8;
}

.system-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    margin-top: 22px;
    padding: 7px 13px;

    border-radius: 999px;

    background: rgba(34, 197, 94, 0.07);
    border: 1px solid rgba(34, 197, 94, 0.18);

    color: #86efac;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 10px rgba(74, 222, 128, 0.7);
}


/* =========================================================
   SECTION LABELS
   ========================================================= */

.section-label {
    font-family: 'Space Grotesk', sans-serif;

    color: #f1f5f9;
    font-size: 1.05rem;
    font-weight: 600;

    margin-bottom: 5px;
}

.section-description {
    color: #717a8a;
    font-size: 0.85rem;
    margin-bottom: 18px;
}


/* =========================================================
   ANALYSIS CARD
   ========================================================= */

.analysis-card {
    background:
        linear-gradient(
            145deg,
            rgba(24, 28, 37, 0.96),
            rgba(17, 20, 27, 0.96)
        );

    border: 1px solid #252a34;
    border-radius: 20px;

    padding: 34px;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.22);
}

.analysis-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    text-align: center;
}

.analysis-card-subtitle {
    color: #737d8d;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 7px;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

div[data-testid="stFileUploader"] {
    margin-top: 24px;
}

div[data-testid="stFileUploader"] section {
    background: rgba(11, 13, 17, 0.65) !important;

    border: 1.5px dashed #343a47 !important;
    border-radius: 15px !important;

    min-height: 150px;

    transition: all 0.2s ease;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #6366f1 !important;
    background: rgba(99, 102, 241, 0.035) !important;
}

div[data-testid="stFileUploader"] label {
    color: #aeb6c4 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    width: 100%;

    min-height: 46px;

    border-radius: 11px !important;
    border: 1px solid #303644 !important;

    background: #171b23 !important;
    color: #e5e7eb !important;

    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;

    transition:
        background 0.18s ease,
        border-color 0.18s ease,
        transform 0.18s ease;
}

.stButton > button:hover {
    background: #202532 !important;
    border-color: #6366f1 !important;
    transform: translateY(-1px);
}

.primary-button .stButton > button {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}


/* =========================================================
   SAMPLE BUTTONS
   ========================================================= */

.sample-label {
    text-align: center;
    color: #687283;

    font-size: 0.75rem;
    font-weight: 600;

    text-transform: uppercase;
    letter-spacing: 0.10em;

    margin: 24px 0 12px 0;
}


/* =========================================================
   DIVIDER
   ========================================================= */

.soft-divider {
    height: 1px;
    background: #20242d;
    margin: 34px 0;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    border-radius: 20px;
    padding: 34px;

    margin-top: 28px;

    text-align: center;
}

.result-card.real {
    background:
        linear-gradient(
            145deg,
            rgba(22, 101, 52, 0.14),
            rgba(17, 24, 39, 0.92)
        );

    border: 1px solid rgba(74, 222, 128, 0.22);
}

.result-card.ai {
    background:
        linear-gradient(
            145deg,
            rgba(127, 29, 29, 0.15),
            rgba(17, 24, 39, 0.92)
        );

    border: 1px solid rgba(248, 113, 113, 0.23);
}

.result-kicker {
    color: #7d8797;
    font-size: 0.72rem;

    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;

    margin-bottom: 10px;
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;

    font-size: clamp(1.8rem, 4vw, 2.6rem);
    font-weight: 700;

    letter-spacing: -0.035em;

    margin: 0;
}

.result-card.real .result-title {
    color: #86efac;
}

.result-card.ai .result-title {
    color: #fca5a5;
}

.confidence-number {
    font-family: 'Space Grotesk', sans-serif;

    font-size: 3.2rem;
    font-weight: 700;

    color: #f8fafc;

    margin-top: 14px;
}

.confidence-label {
    color: #717b8c;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
}


/* =========================================================
   CONFIDENCE BAR
   ========================================================= */

.confidence-wrap {
    max-width: 600px;
    margin: 24px auto 0 auto;
    text-align: left;
}

.confidence-row {
    display: flex;
    justify-content: space-between;

    color: #8992a2;
    font-size: 0.76rem;

    margin-bottom: 7px;
}

.confidence-track {
    height: 7px;

    background: #252a34;
    border-radius: 999px;

    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
}

.confidence-fill.real {
    background: #4ade80;
}

.confidence-fill.ai {
    background: #f87171;
}


/* =========================================================
   AUDIO INFORMATION
   ========================================================= */

.audio-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;

    margin: 12px 0 15px 0;
}

.audio-chip {
    display: inline-flex;

    padding: 6px 10px;

    background: #151922;
    border: 1px solid #272d38;

    border-radius: 8px;

    color: #8d97a7;
    font-size: 0.73rem;
}


/* =========================================================
   FORENSICS CARD
   ========================================================= */

.forensics-card {
    background: #11141a;

    border: 1px solid #252a34;
    border-radius: 18px;

    padding: 22px 24px;

    margin-top: 24px;
}

.forensics-title {
    font-family: 'Space Grotesk', sans-serif;

    color: #f1f5f9;
    font-size: 1.05rem;
    font-weight: 600;

    margin-bottom: 2px;
}

.forensics-subtitle {
    color: #6f7888;
    font-size: 0.78rem;

    margin-bottom: 15px;
}


/* =========================================================
   EXPLAINABILITY
   ========================================================= */

.explain-card {
    background: #11141a;

    border: 1px solid #252a34;
    border-radius: 18px;

    padding: 22px 24px;

    margin-top: 24px;
}

.explain-title {
    font-family: 'Space Grotesk', sans-serif;

    color: #f1f5f9;
    font-size: 1.05rem;
    font-weight: 600;
}

.explain-note {
    color: #697384;
    font-size: 0.76rem;
    margin-top: 4px;
}


/* =========================================================
   TECHNICAL DETAILS
   ========================================================= */

.tech-card {
    background: #101319;

    border: 1px solid #20252e;
    border-radius: 14px;

    padding: 18px 20px;

    margin-top: 18px;
}

.tech-item {
    display: flex;
    justify-content: space-between;

    padding: 9px 0;

    border-bottom: 1px solid #1d222b;

    font-size: 0.80rem;
}

.tech-item:last-child {
    border-bottom: none;
}

.tech-key {
    color: #697384;
}

.tech-value {
    color: #cbd5e1;
    font-weight: 500;
}


/* =========================================================
   HISTORY
   ========================================================= */

.history-card {
    background: #11141a;

    border: 1px solid #252a34;
    border-radius: 16px;

    padding: 20px;
}


/* =========================================================
   STREAMLIT ELEMENT OVERRIDES
   ========================================================= */

.stAudio {
    margin-top: 10px;
}

[data-testid="stExpander"] {
    background: #11141a;
    border: 1px solid #252a34;
    border-radius: 14px;
}

[data-testid="stExpander"] summary {
    color: #cbd5e1 !important;
}

.stAlert {
    border-radius: 12px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2rem;
    }

    .hero h1 {
        font-size: 2.7rem;
    }

    .analysis-card {
        padding: 22px;
    }

    .confidence-number {
        font-size: 2.6rem;
    }

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

if "processed" not in st.session_state:
    st.session_state.processed = set()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">
        🎙️
    </div>

    <h1>
        AI Voice<br>
        <span class="accent">Cloning Detector</span>
    </h1>

    <div class="hero-subtitle">
        Detect AI-generated and cloned speech using
        machine-learning powered audio analysis.
    </div>

    <div class="system-status">
        <span class="status-dot"></span>
        System Ready
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown("""
<div class="section-label">Analyze an audio sample</div>
<div class="section-description">
Upload an audio file, record your voice, or test the detector with a sample.
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Sample Tests",
    "Live Recording",
    "Upload Audio"
])


# ============================================================
# AUDIO HASH
# ============================================================

def get_bytes_hash(data):
    return hashlib.md5(data).hexdigest()


# ============================================================
# WAVEFORM
# ============================================================

def show_waveform(audio, sr):

    fig, ax = plt.subplots(figsize=(10, 2.0))

    time = np.arange(len(audio)) / sr

    ax.plot(
        time,
        audio,
        linewidth=0.8
    )

    ax.set_xlim(0, time[-1] if len(time) else 1)

    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)

    ax.grid(
        True,
        alpha=0.08,
        linewidth=0.6
    )

    for spine in ax.spines.values():
        spine.set_color("#252a34")

    fig.tight_layout(pad=1.0)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(audio, sr):

    fig, ax = plt.subplots(figsize=(10, 3.3))

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

    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)

    ax.tick_params(
        axis="both",
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

    fig.tight_layout(pad=1.0)

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

    importances = np.asarray(
        model.feature_importances_
    )

    feature_names = [
        f"MFCC-{i + 1}"
        for i in range(len(importances))
    ]

    # Top 6 features only
    order = np.argsort(importances)[::-1][:6]

    selected_importances = importances[order][::-1]
    selected_names = np.array(feature_names)[order][::-1]

    fig, ax = plt.subplots(
        figsize=(8, 2.8)
    )

    ax.barh(
        selected_names,
        selected_importances,
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

    for spine in ax.spines.values():
        spine.set_color("#252a34")

    fig.tight_layout(
        pad=1.0
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# RESULT CARD
# ============================================================

def show_result_card(
    result_label,
    confidence_value
):

    if result_label == "Real Voice":

        st.markdown(
            f"""
            <div class="result-card real">

                <div class="result-kicker">
                    Analysis Complete
                </div>

                <div class="result-title">
                    ✓ Human Voice
                </div>

                <div class="confidence-number">
                    {confidence_value:.1f}%
                </div>

                <div class="confidence-label">
                    Detection Confidence
                </div>

                <div class="confidence-wrap">

                    <div class="confidence-row">
                        <span>Human voice</span>
                        <span>{confidence_value:.1f}%</span>
                    </div>

                    <div class="confidence-track">
                        <div
                            class="confidence-fill real"
                            style="width:{confidence_value}%"
                        ></div>
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-card ai">

                <div class="result-kicker">
                    Analysis Complete
                </div>

                <div class="result-title">
                    ⚠ AI-Generated Voice
                </div>

                <div class="confidence-number">
                    {confidence_value:.1f}%
                </div>

                <div class="confidence-label">
                    Detection Confidence
                </div>

                <div class="confidence-wrap">

                    <div class="confidence-row">
                        <span>AI-generated voice</span>
                        <span>{confidence_value:.1f}%</span>
                    </div>

                    <div class="confidence-track">
                        <div
                            class="confidence-fill ai"
                            style="width:{confidence_value}%"
                        ></div>
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def predict_audio(
    load_path,
    display_name,
    show_audio=True,
    unique_id=None
):

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        audio, sr = librosa.load(
            load_path,
            sr=16000
        )

        duration = len(audio) / sr

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if duration < 0.5:

            st.warning(
                f"Audio is too short. Please provide at least 0.5 seconds."
            )
            return

        if np.max(np.abs(audio)) < 0.01:

            st.warning(
                "This audio appears to be silent or has very low volume."
            )
            return

        # ----------------------------------------------------
        # Feature extraction
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
        # Prediction
        # ----------------------------------------------------

        pred = model.predict(
            mfcc_mean
        )[0]

        confidence = model.predict_proba(
            mfcc_mean
        )[0]

        # Your existing model mapping:
        # 1 = Real Voice
        # 0 = AI-Cloned Voice

        if pred == 1:

            result_label = "Real Voice"
            conf_value = float(
                confidence[1] * 100
            )

        else:

            result_label = "AI-Cloned Voice"
            conf_value = float(
                confidence[0] * 100
            )

        # ----------------------------------------------------
        # Prevent duplicate history entries
        # ----------------------------------------------------

        history_key = unique_id or f"{display_name}_{result_label}"

        if history_key not in st.session_state.processed:

            st.session_state.history.append({
                "Audio": display_name,
                "Result": result_label,
                "Confidence": f"{conf_value:.1f}%"
            })

            st.session_state.processed.add(
                history_key
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if show_audio:

            show_result_card(
                result_label,
                conf_value
            )

            # ------------------------------------------------
            # Audio metadata
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="audio-meta">

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
                        Mono
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Audio player
            # ------------------------------------------------

            st.audio(
                load_path
            )

            # ------------------------------------------------
            # AUDIO FORENSICS
            # ------------------------------------------------

            st.markdown(
                """
                <div class="forensics-card">

                    <div class="forensics-title">
                        Audio Forensics
                    </div>

                    <div class="forensics-subtitle">
                        Visual representation of the analyzed audio signal
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "**Waveform**"
            )

            show_waveform(
                audio,
                sr
            )

            st.markdown(
                "**Spectrogram**"
            )

            show_spectrogram(
                audio,
                sr
            )

            # ------------------------------------------------
            # EXPLAINABILITY
            # ------------------------------------------------

            st.markdown(
                """
                <div class="explain-card">

                    <div class="explain-title">
                        🔍 Model Explainability
                    </div>

                    <div class="explain-note">
                        Relative importance of MFCC features used by the trained model.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            show_feature_importance()

        return result_label, conf_value

    except Exception as e:

        st.error(
            f"Error processing {display_name}: {str(e)}"
        )

        return None, None


# ============================================================
# SAMPLE TEST TAB
# ============================================================

with tab1:

    st.markdown(
        """
        <div class="analysis-card">

            <div class="analysis-card-title">
                Test the detector
            </div>

            <div class="analysis-card-subtitle">
                Use the built-in samples to see how the system distinguishes
                human speech from AI-generated speech.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sample-label">Choose a sample</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        if st.button(
            "✓  Test Real Voice",
            key="real_sample",
            use_container_width=True
        ):

            predict_audio(
                "sample_real.flac",
                "sample_real.flac",
                show_audio=True,
                unique_id="sample_real"
            )

    with col2:

        if st.button(
            "⚠  Test AI-Cloned Voice",
            key="fake_sample",
            use_container_width=True
        ):

            predict_audio(
                "sample_fake.flac",
                "sample_fake.flac",
                show_audio=True,
                unique_id="sample_fake"
            )


# ============================================================
# LIVE RECORD TAB
# ============================================================

with tab2:

    st.markdown(
        """
        <div class="analysis-card">

            <div class="analysis-card-title">
                Record your voice
            </div>

            <div class="analysis-card-subtitle">
                Record a short speech sample and let the detector analyze it.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    mic_input = st.audio_input(
        "Record using your microphone",
        key="microphone_input"
    )

    if mic_input is not None:

        mic_bytes = mic_input.getvalue()

        mic_hash = get_bytes_hash(
            mic_bytes
        )

        temp_path = "temp_mic.wav"

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                mic_bytes
            )

        predict_audio(
            temp_path,
            "Microphone Recording",
            show_audio=True,
            unique_id=f"mic_{mic_hash}"
        )


# ============================================================
# UPLOAD TAB
# ============================================================

with tab3:

    st.markdown(
        """
        <div class="analysis-card">

            <div class="analysis-card-title">
                Analyze your own audio
            </div>

            <div class="analysis-card-subtitle">
                Upload WAV or FLAC speech recordings for authenticity analysis.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Drop audio files here",
        type=["flac", "wav"],
        accept_multiple_files=True,
        help="Supported formats: WAV and FLAC"
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()

            file_hash = get_bytes_hash(
                file_bytes
            )

            temp_input = (
                f"temp_{file_hash}.wav"
            )

            with open(
                temp_input,
                "wb"
            ) as f:

                f.write(
                    file_bytes
                )

            predict_audio(
                temp_input,
                uploaded_file.name,
                show_audio=True,
                unique_id=f"upload_{file_hash}"
            )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.markdown(
    '<div class="soft-divider"></div>',
    unsafe_allow_html=True
)

with st.expander(
    "Technical Details"
):

    st.markdown(
        """
        <div class="tech-card">

            <div class="tech-item">
                <span class="tech-key">Model accuracy</span>
                <span class="tech-value">90%</span>
            </div>

            <div class="tech-item">
                <span class="tech-key">Training samples</span>
                <span class="tech-value">5,160</span>
            </div>

            <div class="tech-item">
                <span class="tech-key">Audio sample rate</span>
                <span class="tech-value">16 kHz</span>
            </div>

            <div class="tech-item">
                <span class="tech-key">Feature representation</span>
                <span class="tech-value">13 MFCC coefficients</span>
            </div>

            <div class="tech-item">
                <span class="tech-key">Supported audio</span>
                <span class="tech-value">WAV / FLAC</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-label">
            Prediction History
        </div>

        <div class="section-description">
            Previous audio analyses from this session.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )
