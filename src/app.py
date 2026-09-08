import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os
import tempfile
import time
from collections import deque

# WebRTC for live microphone analysis
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Voice Shield",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MATPLOTLIB
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
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d0f14;
    color: #e5e7eb;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}


/* ================= HEADER ================= */

.hero {
    text-align: center;
    padding: 20px 0 28px 0;
}

.hero-icon {
    font-size: 2.6rem;
    margin-bottom: 4px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.045em;
    color: #f8fafc;
    line-height: 1.05;
}

.hero-title span {
    color: #818cf8;
}

.hero-subtitle {
    color: #8f98a8;
    font-size: 0.95rem;
    margin-top: 12px;
}

.status-line {
    margin-top: 15px;
    color: #86efac;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
}


/* ================= SECTION ================= */

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-top: 1rem;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #737d8d;
    font-size: 0.78rem;
    margin-bottom: 1.2rem;
}


/* ================= BUTTONS ================= */

.stButton > button {
    width: 100%;
    min-height: 46px;

    background: #171b23;
    color: #e5e7eb;

    border: 1px solid #303641;
    border-radius: 10px;

    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.86rem;

    transition: all 0.18s ease;
}

.stButton > button:hover {
    background: #202532;
    border-color: #6366f1;
    color: white;
}


/* ================= TABS ================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #252a34;
}

.stTabs [data-baseweb="tab"] {
    color: #858e9e;
    font-size: 0.86rem;
    font-weight: 500;
    padding: 12px 17px;
}

.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
}


/* ================= RESULT ================= */

.risk-box {
    border-radius: 14px;
    padding: 26px;
    text-align: center;
    margin: 1rem 0;
}

.low-risk {
    background: #101b16;
    border: 1px solid #24543b;
}

.medium-risk {
    background: #1b1910;
    border: 1px solid #63531d;
}

.high-risk {
    background: #1c1215;
    border: 1px solid #632b34;
}

.risk-small {
    color: #7e8796;
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    font-weight: 600;
}

.risk-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.1rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
    margin-top: 5px;
}

.risk-status {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 5px;
}

.low-text {
    color: #86efac;
}

.medium-text {
    color: #fde68a;
}

.high-text {
    color: #fca5a5;
}

.risk-description {
    color: #8b95a5;
    font-size: 0.78rem;
    margin-top: 8px;
}


/* ================= CHIPS ================= */

.chip-row {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    margin: 12px 0;
}

.chip {
    background: #151922;
    border: 1px solid #292f3a;
    border-radius: 7px;
    padding: 5px 9px;
    color: #8e98a8;
    font-size: 0.7rem;
}


/* ================= RECOMMENDATION ================= */

.recommendation {
    background: #12161d;
    border: 1px solid #292f39;
    border-left: 3px solid #6366f1;
    border-radius: 9px;
    padding: 15px 17px;
    margin: 12px 0;
}

.recommendation-title {
    color: #c7d2fe;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 4px;
}

.recommendation-text {
    color: #9ba4b3;
    font-size: 0.77rem;
}


/* ================= FILE UPLOADER ================= */

div[data-testid="stFileUploader"] section {
    background: #11141a !important;
    border: 1.5px dashed #353c49 !important;
    border-radius: 12px !important;
}


/* ================= AUDIO ================= */

audio {
    width: 100%;
}


/* ================= LIVE STATUS ================= */

.live-card {
    background: #11151c;
    border: 1px solid #29303b;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
}

.live-label {
    color: #747e8e;
    font-size: 0.67rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.live-value {
    color: #e5e7eb;
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 4px;
}


/* ================= TECHNICAL ================= */

.tech-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 2px;
    border-bottom: 1px solid #20252e;
    font-size: 0.78rem;
}

.tech-name {
    color: #737d8d;
}

.tech-value {
    color: #cbd5e1;
    font-weight: 500;
}


/* ================= DIVIDER ================= */

hr {
    border-color: #242933 !important;
    margin: 2rem 0 !important;
}


/* ================= EXPANDER ================= */

.streamlit-expanderHeader {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# MODEL
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

if "live_scores" not in st.session_state:
    st.session_state.live_scores = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

    <div class="hero-icon">🎙️</div>

    <div class="hero-title">
        Voice <span>Shield</span>
    </div>

    <div class="hero-subtitle">
        AI voice cloning and impersonation risk detection
    </div>

    <div class="status-line">
        ● DETECTION ENGINE ONLINE
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPER: RISK
# ============================================================

def calculate_risk(ai_probability):

    risk = float(ai_probability)

    if risk < 30:
        level = "LOW RISK"
        css = "low-risk"
        text = "low-text"
        icon = "🟢"
        description = (
            "No strong indicators of synthetic speech were detected."
        )
        recommendation = (
            "Normal interaction can continue. "
            "Standard verification practices are still recommended."
        )

    elif risk < 70:
        level = "SUSPICIOUS"
        css = "medium-risk"
        text = "medium-text"
        icon = "🟡"
        description = (
            "The audio contains characteristics that warrant caution."
        )
        recommendation = (
            "Exercise caution and verify the caller through "
            "a trusted secondary channel before sensitive actions."
        )

    else:
        level = "HIGH RISK"
        css = "high-risk"
        text = "high-text"
        icon = "🔴"
        description = (
            "Strong indicators of AI-generated or manipulated speech "
            "were detected."
        )
        recommendation = (
            "Do not authorize sensitive actions based on this call alone. "
            "Use callback verification, MFA, or escalation."
        )

    return {
        "risk": risk,
        "level": level,
        "css": css,
        "text": text,
        "icon": icon,
        "description": description,
        "recommendation": recommendation
    }


# ============================================================
# RESULT UI
# ============================================================

def display_risk_result(ai_probability, show_recommendation=True):

    result = calculate_risk(ai_probability)

    st.markdown(
        f"""
<div class="risk-box {result['css']}">

    <div class="risk-small">
        Voice Integrity Assessment
    </div>

    <div class="risk-number">
        {result['risk']:.0f}
    </div>

    <div class="risk-status {result['text']}">
        {result['icon']} {result['level']}
    </div>

    <div class="risk-description">
        {result['description']}
    </div>

</div>
""",
        unsafe_allow_html=True
    )

    st.progress(
        min(max(result["risk"] / 100, 0.0), 1.0)
    )

    if show_recommendation:

        st.markdown(
            f"""
<div class="recommendation">

    <div class="recommendation-title">
        Recommended Action
    </div>

    <div class="recommendation-text">
        {result['recommendation']}
    </div>

</div>
""",
            unsafe_allow_html=True
        )

    return result


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
            "Feature importance is not available for this classifier."
        )

        return

    importances = model.feature_importances_

    feature_names = [
        f"MFCC-{i + 1}"
        for i in range(len(importances))
    ]

    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 3))

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
# MODEL PREDICTION
# ============================================================

def analyze_array(audio, sr=16000):

    if len(audio) == 0:
        return None

    duration = len(audio) / sr

    if duration < 0.4:
        return None

    peak = np.max(
        np.abs(audio)
    )

    if peak < 0.005:
        return None

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    ).reshape(1, -1)

    pred = model.predict(
        mfcc_mean
    )[0]

    probabilities = model.predict_proba(
        mfcc_mean
    )[0]

    if pred == 1:

        real_probability = probabilities[1]
        ai_probability = probabilities[0]

    else:

        real_probability = probabilities[1]
        ai_probability = probabilities[0]

    return {
        "prediction": pred,
        "ai_probability": float(ai_probability),
        "real_probability": float(real_probability),
        "duration": duration,
        "mfcc": mfcc_mean
    }


# ============================================================
# FILE ANALYSIS
# ============================================================

def predict_audio(
    load_path,
    display_name,
    show_details=True
):

    try:

        audio, sr = librosa.load(
            load_path,
            sr=16000
        )

        result = analyze_array(
            audio,
            sr
        )

        if result is None:

            st.warning(
                "⚠️ Audio is too short, silent, or could not be analyzed."
            )

            return

        ai_probability = (
            result["ai_probability"] * 100
        )

        # --------------------------------------------
        # RESULT
        # --------------------------------------------

        display_risk_result(
            ai_probability
        )

        # --------------------------------------------
        # INFORMATION CHIPS
        # --------------------------------------------

        st.markdown(
            f"""
<div class="chip-row">

    <span class="chip">
        🎵 {display_name}
    </span>

    <span class="chip">
        ⏱ {result['duration']:.1f} sec
    </span>

    <span class="chip">
        16 kHz
    </span>

    <span class="chip">
        13 MFCC
    </span>

</div>
""",
            unsafe_allow_html=True
        )

        st.audio(
            load_path
        )

        if show_details:

            # ----------------------------------------
            # FORENSICS
            # ----------------------------------------

            st.markdown(
                """
<div class="section-title">
    Audio Forensics
</div>

<div class="section-subtitle">
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

            # ----------------------------------------
            # EXPLAINABILITY
            # ----------------------------------------

            with st.expander(
                "🔍 Model Explainability"
            ):

                st.markdown(
                    """
The classifier uses 13 Mel-Frequency Cepstral
Coefficients (MFCCs) to represent the acoustic
characteristics of the speech signal.

The chart shows the relative feature importance
reported by the trained classifier.
"""
                )

                show_feature_importance()

        # --------------------------------------------
        # HISTORY
        # --------------------------------------------

        risk = calculate_risk(
            ai_probability
        )

        st.session_state.history.append(
            {
                "Audio": display_name,
                "Assessment": risk["level"],
                "AI Probability": f"{ai_probability:.1f}%"
            }
        )

    except Exception as e:

        st.error(
            f"Error processing {display_name}: {str(e)}"
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

st.markdown(
    """
<div class="section-title">
    Analyze an audio sample
</div>

<div class="section-subtitle">
    Test a known sample, monitor microphone input,
    or analyze your own recording.
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
        "🎤 Live Voice Analysis",
        "📤 Upload Audio"
    ]
)


# ============================================================
# SAMPLE TAB
# ============================================================

with tab1:

    st.markdown(
        """
<div class="section-title">
    Test the detection engine
</div>

<div class="section-subtitle">
    Use the provided samples to demonstrate both outcomes.
</div>
""",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(
        2,
        gap="large"
    )

    with c1:

        if st.button(
            "✓  Analyze Real Voice",
            key="real_test"
        ):

            predict_audio(
                "sample_real.flac",
                "Real Voice Sample"
            )

    with c2:

        if st.button(
            "⚠  Analyze AI-Cloned Voice",
            key="fake_test"
        ):

            predict_audio(
                "sample_fake.flac",
                "AI-Cloned Voice Sample"
            )


# ============================================================
# LIVE VOICE TAB
# ============================================================

with tab2:

    st.markdown(
        """
<div class="section-title">
    Live Voice Analysis
</div>

<div class="section-subtitle">
    Continuously monitor microphone audio and estimate
    synthetic-voice risk from short audio windows.
</div>
""",
        unsafe_allow_html=True
    )

    if not WEBRTC_AVAILABLE:

        st.error(
            "Live microphone analysis requires streamlit-webrtc."
        )

        st.info(
            "Add streamlit-webrtc to requirements.txt and restart the app."
        )

    else:

        # ----------------------------------------------------
        # LIVE PROCESSOR
        # ----------------------------------------------------

        class AudioProcessor:

            def __init__(self):

                self.buffer = np.array(
                    [],
                    dtype=np.float32
                )

                self.latest_score = 0.0
                self.history = deque(
                    maxlen=20
                )

                self.sample_rate = 16000

            def recv(self, frame):

                audio = frame.to_ndarray()

                # Handle stereo
                if audio.ndim > 1:
                    audio = np.mean(
                        audio,
                        axis=0
                    )

                audio = audio.astype(
                    np.float32
                )

                # Normalize common int formats
                if np.max(
                    np.abs(audio)
                ) > 1.5:

                    audio = audio / 32768.0

                self.buffer = np.concatenate(
                    [
                        self.buffer,
                        audio
                    ]
                )

                # ----------------------------------------
                # Analyze approximately every 1.5 sec
                # ----------------------------------------

                required_samples = int(
                    self.sample_rate * 1.5
                )

                if len(
                    self.buffer
                ) >= required_samples:

                    chunk = self.buffer[
                        :required_samples
                    ]

                    self.buffer = self.buffer[
                        required_samples // 2:
                    ]

                    try:

                        result = analyze_array(
                            chunk,
                            self.sample_rate
                        )

                        if result is not None:

                            score = (
                                result["ai_probability"]
                                * 100
                            )

                            self.latest_score = score

                            self.history.append(
                                (
                                    time.time(),
                                    score
                                )
                            )

                    except Exception:
                        pass

                return frame


        # ----------------------------------------------------
        # WEBRTC
        # ----------------------------------------------------

        ctx = webrtc_streamer(
            key="voice-shield-live",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={
                "video": False,
                "audio": True
            },
            async_processing=True
        )

        # ----------------------------------------------------
        # LIVE DISPLAY
        # ----------------------------------------------------

        if ctx.audio_processor:

            processor = ctx.audio_processor

            st.markdown(
                """
<div class="live-card">

    <div class="live-label">
        Monitoring Status
    </div>

    <div class="live-value">
        🎙️ Microphone stream connected
    </div>

</div>
""",
                unsafe_allow_html=True
            )

            score = processor.latest_score

            if score > 0:

                result = calculate_risk(
                    score
                )

                st.markdown(
                    f"""
<div class="risk-box {result['css']}">

    <div class="risk-small">
        Live Voice Risk
    </div>

    <div class="risk-number">
        {score:.0f}
    </div>

    <div class="risk-status {result['text']}">
        {result['icon']} {result['level']}
    </div>

    <div class="risk-description">
        {result['description']}
    </div>

</div>
""",
                    unsafe_allow_html=True
                )

                st.progress(
                    min(
                        max(score / 100, 0),
                        1
                    )
                )

                st.markdown(
                    f"""
<div class="recommendation">

    <div class="recommendation-title">
        Recommended Action
    </div>

    <div class="recommendation-text">
        {result['recommendation']}
    </div>

</div>
""",
                    unsafe_allow_html=True
                )

                # ----------------------------------------
                # TIMELINE
                # ----------------------------------------

                st.markdown(
                    """
<div class="section-title">
    Live Risk Timeline
</div>

<div class="section-subtitle">
    Risk estimates generated from successive audio windows.
</div>
""",
                    unsafe_allow_html=True
                )

                scores = [
                    score_value
                    for _, score_value
                    in processor.history
                ]

                if scores:

                    fig, ax = plt.subplots(
                        figsize=(8, 2.4)
                    )

                    ax.plot(
                        range(
                            1,
                            len(scores) + 1
                        ),
                        scores,
                        marker="o",
                        linewidth=2
                    )

                    ax.axhline(
                        70,
                        linestyle="--",
                        alpha=0.35
                    )

                    ax.axhline(
                        30,
                        linestyle="--",
                        alpha=0.25
                    )

                    ax.set_ylim(
                        0,
                        100
                    )

                    ax.set_ylabel(
                        "Risk",
                        fontsize=8
                    )

                    ax.set_xlabel(
                        "Analysis window",
                        fontsize=8
                    )

                    ax.tick_params(
                        labelsize=8
                    )

                    ax.grid(
                        alpha=0.08
                    )

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    plt.close(fig)

            else:

                st.info(
                    "Start the microphone and speak for a few seconds. "
                    "The detector will begin evaluating audio windows."
                )


# ============================================================
# UPLOAD TAB
# ============================================================

with tab3:

    st.markdown(
        """
<div class="section-title">
    Upload Audio
</div>

<div class="section-subtitle">
    Analyze WAV or FLAC recordings for synthetic voice indicators.
</div>
""",
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Drop audio files here",
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
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = tmp.name

            predict_audio(
                temp_path,
                uploaded_file.name,
                show_details=True
            )

            try:
                os.remove(
                    temp_path
                )
            except:
                pass


# ============================================================
# PRIVACY + TECHNICAL DETAILS
# ============================================================

st.markdown("---")

c1, c2 = st.columns(
    2,
    gap="large"
)

with c1:

    with st.expander(
        "🔐 Privacy & Processing"
    ):

        st.markdown(
            """
            <div class="tech-row">
                <span class="tech-name">Raw audio retention</span>
                <span class="tech-value">Session-based</span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Inference</span>
                <span class="tech-value">ML classifier</span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Feature representation</span>
                <span class="tech-value">MFCC</span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Audio sample rate</span>
                <span class="tech-value">16 kHz</span>
            </div>
            """,
            unsafe_allow_html=True
        )

with c2:

    with st.expander(
        "⚙️ Technical Details"
    ):

        st.markdown(
            """
            <div class="tech-row">
                <span class="tech-name">Classifier</span>
                <span class="tech-value">
                    Trained voice classification model
                </span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Features</span>
                <span class="tech-value">
                    13 MFCC coefficients
                </span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Training samples</span>
                <span class="tech-value">
                    5,160
                </span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Reported accuracy</span>
                <span class="tech-value">
                    90%
                </span>
            </div>

            <div class="tech-row">
                <span class="tech-name">Supported formats</span>
                <span class="tech-value">
                    WAV / FLAC
                </span>
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
    Results generated during the current session.
</div>
""",
        unsafe_allow_html=True
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )
