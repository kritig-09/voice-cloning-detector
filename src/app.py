import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
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
# MATPLOTLIB CONFIG
# ============================================================

matplotlib.rcParams.update({
    "figure.facecolor": "#0b0d12",
    "axes.facecolor": "#0b0d12",
    "axes.edgecolor": "#30343d",
    "axes.labelcolor": "#c9ced8",
    "text.color": "#e6e8ed",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af",
    "font.size": 10,
})

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 50% -10%, #171b2b 0%, #0b0d12 42%),
        #0b0d12;
    color: #e6e8ed;
}

/* Remove excessive Streamlit spacing */

.block-container {
    max-width: 1180px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* =========================================================
   HERO
   ========================================================= */

.hero {
    text-align: center;
    padding: 30px 20px 26px 20px;
    margin-bottom: 25px;
}

.hero-icon {
    font-size: 42px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #f8fafc;
    line-height: 1.15;
}

.hero-title span {
    color: #818cf8;
}

.hero-subtitle {
    margin-top: 12px;
    font-size: 1rem;
    color: #9da4b2;
    font-weight: 500;
}

.status-line {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 18px;
    padding: 7px 14px;
    border: 1px solid #28303c;
    border-radius: 999px;
    background: #11141b;
    color: #aeb6c4;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.7px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    display: inline-block;
    box-shadow: 0 0 10px rgba(52, 211, 153, 0.6);
}

/* =========================================================
   INFO STRIP
   ========================================================= */

.info-strip {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin: 5px 0 28px 0;
}

.info-chip {
    border: 1px solid #282e39;
    background: #11141b;
    border-radius: 10px;
    padding: 8px 14px;
    color: #aeb6c4;
    font-size: 0.78rem;
    font-weight: 600;
}

/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-title {
    font-size: 1.25rem;
    font-weight: 750;
    color: #f1f3f6;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-description {
    color: #8f98a8;
    font-size: 0.88rem;
    margin-bottom: 18px;
}

/* =========================================================
   TABS
   ========================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #242933;
}

.stTabs [data-baseweb="tab"] {
    color: #929aaa;
    font-size: 0.9rem;
    font-weight: 650;
    padding: 12px 18px;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #d8dce5;
}

.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
}

/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 10px;
    border: 1px solid #343b49;
    background: #151922;
    color: #e9ebef;
    font-family: 'Manrope', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6366f1;
    background: #1a1e2a;
    color: white;
}

/* =========================================================
   UPLOADER
   ========================================================= */

div[data-testid="stFileUploader"] {
    background: #11141b;
    border: 1px dashed #3a4250;
    border-radius: 12px;
    padding: 8px;
}

/* =========================================================
   AUDIO
   ========================================================= */

audio {
    width: 100%;
    border-radius: 10px;
}

/* =========================================================
   ANALYSIS CARDS
   ========================================================= */

.analysis-card {
    background: #11141b;
    border: 1px solid #282e39;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}

.card-label {
    color: #8e97a7;
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.card-value {
    color: #f2f4f7;
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 4px;
}

/* =========================================================
   RESULT
   ========================================================= */

.result-card {
    border: 1px solid #303746;
    border-radius: 16px;
    padding: 25px;
    background: #11141b;
    margin: 22px 0;
    text-align: center;
}

.result-title {
    color: #8e97a7;
    font-size: 0.74rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.result-main {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 8px;
}

.result-confidence {
    color: #aeb6c4;
    margin-top: 5px;
    font-size: 0.9rem;
}

/* =========================================================
   PROBABILITY BOXES
   ========================================================= */

.probability-box {
    background: #11141b;
    border: 1px solid #282e39;
    border-radius: 12px;
    padding: 17px;
    text-align: center;
}

.probability-label {
    color: #8f98a8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
}

.probability-value {
    color: #e8ebef;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 4px;
}

/* =========================================================
   RISK
   ========================================================= */

.risk-high {
    color: #fb7185;
}

.risk-medium {
    color: #fbbf24;
}

.risk-low {
    color: #34d399;
}

.risk-box {
    border-radius: 12px;
    padding: 15px 18px;
    margin: 15px 0;
    background: #11141b;
    border: 1px solid #303746;
}

.risk-title {
    font-size: 1rem;
    font-weight: 800;
}

.risk-text {
    color: #9da5b3;
    font-size: 0.83rem;
    margin-top: 5px;
}

/* =========================================================
   FORENSICS
   ========================================================= */

.forensics-card {
    background: #11141b;
    border: 1px solid #282e39;
    border-radius: 14px;
    padding: 20px;
    margin-top: 15px;
}

.forensic-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid #20242d;
}

.forensic-row:last-child {
    border-bottom: none;
}

.forensic-name {
    color: #8f98a8;
    font-size: 0.82rem;
}

.forensic-value {
    color: #e2e5ea;
    font-size: 0.82rem;
    font-weight: 700;
}

/* =========================================================
   NOTICE
   ========================================================= */

.notice {
    background: #10141c;
    border: 1px solid #2d3440;
    border-radius: 12px;
    padding: 15px 18px;
    color: #aeb6c4;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 18px;
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    margin-top: 55px;
    padding-top: 20px;
    border-top: 1px solid #20242c;
    color: #656e7d;
    font-size: 0.75rem;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("voice_cloning_model.pkl")
except Exception as e:
    st.error("Unable to load the trained voice detection model.")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(ai_probability):
    """
    Converts model AI probability into an easy-to-understand
    risk classification.
    """

    if ai_probability >= 70:
        return "HIGH RISK", "risk-high", \
               "Strong model indication of AI-generated or cloned speech."

    elif ai_probability >= 30:
        return "SUSPICIOUS", "risk-medium", \
               "The model detected characteristics that require additional verification."

    else:
        return "LOW RISK", "risk-low", \
               "The model found stronger characteristics associated with human speech."


def check_audio_quality(audio, sr):
    """
    Basic signal-quality checks.
    """

    duration = len(audio) / sr if sr else 0

    peak = float(np.max(np.abs(audio))) if len(audio) else 0

    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0

    if peak < 0.01:
        level = "Very Low"
    elif peak < 0.08:
        level = "Low"
    elif peak < 0.8:
        level = "Good"
    else:
        level = "High"

    if rms < 0.005:
        signal_quality = "Very Low"
    elif rms < 0.02:
        signal_quality = "Low"
    else:
        signal_quality = "Good"

    duration_status = "Good" if duration >= 1 else "Too Short"

    return {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "level": level,
        "signal_quality": signal_quality,
        "duration_status": duration_status
    }


def show_spectrogram(audio, sr, title):
    """
    Displays an audio spectrogram.
    """

    fig, ax = plt.subplots(figsize=(9, 3.2))

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

    ax.set_title(title, fontsize=11, pad=10)
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


def show_feature_importance():
    """
    Displays model feature importance when supported by the
    trained classifier.
    """

    if not hasattr(model, "feature_importances_"):
        st.info(
            "Feature-level importance is not available for this model type."
        )
        return

    importances = np.asarray(model.feature_importances_)

    feature_names = [
        f"MFCC-{i + 1}"
        for i in range(len(importances))
    ]

    # Show most influential features first
    indices = np.argsort(importances)[::-1]

    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 3.4))

    ax.bar(
        sorted_names,
        sorted_values
    )

    ax.set_title(
        "MFCC feature influence",
        fontsize=11,
        pad=10
    )

    ax.set_ylabel("Relative importance")

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=8
    )

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    top_features = sorted_names[:3]

    st.caption(
        "Most influential features: "
        + ", ".join(top_features)
    )


def add_to_history(display_name, result_label, ai_probability, risk):
    """
    Adds a result to session history while avoiding duplicates
    during Streamlit reruns.
    """

    entry = {
        "Filename": display_name,
        "Result": result_label,
        "AI Probability": f"{ai_probability:.1f}%",
        "Risk": risk
    }

    # Avoid duplicate consecutive entries
    if not st.session_state.history:
        st.session_state.history.append(entry)

    elif st.session_state.history[-1] != entry:
        st.session_state.history.append(entry)


def display_result(
    result_label,
    ai_probability,
    human_probability,
    risk,
    risk_class,
    risk_description
):
    """
    Professional result section.
    """

    if result_label == "AI-Cloned Voice":
        icon = "⚠"
    else:
        icon = "✓"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Voice Authenticity Assessment</div>
            <div class="result-main">
                {icon} {result_label}
            </div>
            <div class="result-confidence">
                Model confidence: {max(ai_probability, human_probability):.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            f"""
            <div class="probability-box">
                <div class="probability-label">AI-generated probability</div>
                <div class="probability-value">{ai_probability:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:
        st.markdown(
            f"""
            <div class="probability-box">
                <div class="probability-label">Human probability</div>
                <div class="probability-value">{human_probability:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="risk-box">
            <div class="risk-title {risk_class}">
                {risk}
            </div>
            <div class="risk-text">
                {risk_description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(
        float(ai_probability / 100),
        text=f"AI-generated probability: {ai_probability:.1f}%"
    )

    if ai_probability >= 70:

        st.markdown(
            """
            <div class="notice">
                <b>Recommended verification:</b><br>
                Do not rely on voice alone for high-risk decisions.
                Verify the speaker through an independent communication
                channel or additional authentication factor.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif ai_probability >= 30:

        st.markdown(
            """
            <div class="notice">
                <b>Verification recommended:</b><br>
                The result is not conclusive. Consider additional
                authentication before trusting the recording.
            </div>
            """,
            unsafe_allow_html=True
        )


def show_forensics(audio, sr, quality):
    """
    Audio technical information.
    """

    st.markdown(
        '<div class="section-title">Audio Forensics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Technical characteristics of the analyzed signal.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="analysis-card">
                <div class="card-label">Duration</div>
                <div class="card-value">{quality["duration"]:.1f}s</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="analysis-card">
                <div class="card-label">Sample Rate</div>
                <div class="card-value">{sr / 1000:.0f} kHz</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="analysis-card">
                <div class="card-label">Channels</div>
                <div class="card-value">Mono</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="analysis-card">
                <div class="card-label">Features</div>
                <div class="card-value">13 MFCC</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="forensics-card">

            <div class="forensic-row">
                <span class="forensic-name">Signal level</span>
                <span class="forensic-value">{quality["level"]}</span>
            </div>

            <div class="forensic-row">
                <span class="forensic-name">Signal quality</span>
                <span class="forensic-value">{quality["signal_quality"]}</span>
            </div>

            <div class="forensic-row">
                <span class="forensic-name">Duration check</span>
                <span class="forensic-value">{quality["duration_status"]}</span>
            </div>

            <div class="forensic-row">
                <span class="forensic-name">Peak amplitude</span>
                <span class="forensic-value">{quality["peak"]:.4f}</span>
            </div>

            <div class="forensic-row">
                <span class="forensic-name">RMS energy</span>
                <span class="forensic-value">{quality["rms"]:.4f}</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def predict_audio(load_path, display_name, show_details=True):
    """
    Complete audio-analysis pipeline.

    Audio
       ↓
    preprocessing
       ↓
    MFCC extraction
       ↓
    ML model
       ↓
    probability
       ↓
    risk assessment
    """

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            load_path,
            sr=16000,
            mono=True
        )

        quality = check_audio_quality(audio, sr)

        duration = quality["duration"]

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if duration < 0.5:

            st.warning(
                f"⚠ {display_name}: Audio is too short. "
                "Please provide at least 0.5 seconds of speech."
            )

            return

        if quality["peak"] < 0.01:

            st.warning(
                f"⚠ {display_name}: Audio appears silent "
                "or extremely low volume."
            )

            return

        # ----------------------------------------------------
        # Audio player
        # ----------------------------------------------------

        if show_details:

            st.audio(
                load_path,
                format="audio/wav"
            )

            st.caption(
                f"Analyzing: {display_name}"
            )

        # ----------------------------------------------------
        # MFCC extraction
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

        pred = model.predict(mfcc_mean)[0]

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                mfcc_mean
            )[0]

            # Your existing model mapping:
            # class 0 = AI
            # class 1 = Real

            try:

                class_indices = list(model.classes_)

                ai_index = class_indices.index(0)
                human_index = class_indices.index(1)

                ai_probability = float(
                    probabilities[ai_index] * 100
                )

                human_probability = float(
                    probabilities[human_index] * 100
                )

            except Exception:

                ai_probability = float(
                    probabilities[0] * 100
                )

                human_probability = float(
                    probabilities[1] * 100
                )

        else:

            if pred == 1:
                human_probability = 100.0
                ai_probability = 0.0
            else:
                human_probability = 0.0
                ai_probability = 100.0

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if pred == 1:

            result_label = "Real Voice"

        else:

            result_label = "AI-Cloned Voice"

        risk, risk_class, risk_description = get_risk_level(
            ai_probability
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        if show_details:

            display_result(
                result_label,
                ai_probability,
                human_probability,
                risk,
                risk_class,
                risk_description
            )

            show_forensics(
                audio,
                sr,
                quality
            )

            # ------------------------------------------------
            # Spectrogram
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Spectral Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Visual representation of the frequency content '
                'of the analyzed speech signal.'
                '</div>',
                unsafe_allow_html=True
            )

            show_spectrogram(
                audio,
                sr,
                f"Speech Spectrogram — {display_name}"
            )

            # ------------------------------------------------
            # Explainability
            # ------------------------------------------------

            with st.expander(
                "🧠 Model Explainability — MFCC Features"
            ):

                st.write(
                    "The classifier uses 13 Mel-Frequency "
                    "Cepstral Coefficients (MFCCs) to represent "
                    "acoustic characteristics of the speech signal."
                )

                show_feature_importance()

            # ------------------------------------------------
            # Verification advice
            # ------------------------------------------------

            with st.expander(
                "🛡️ Recommended Security Response"
            ):

                st.markdown(
                    """
                    **If this voice is being used for a sensitive
                    decision:**

                    1. Do not rely on voice alone.
                    2. Verify the speaker using a trusted contact
                       method.
                    3. Use an additional authentication factor.
                    4. Avoid sharing sensitive information until
                       identity is independently confirmed.
                    """
                )

            # ------------------------------------------------
            # Technical pipeline
            # ------------------------------------------------

            with st.expander(
                "⚙️ Detection Pipeline"
            ):

                st.markdown(
                    """
                    **Audio Input**

                    ↓

                    **16 kHz Audio Normalization**

                    ↓

                    **MFCC Feature Extraction**

                    ↓

                    **13-Dimensional Feature Vector**

                    ↓

                    **Machine-Learning Classifier**

                    ↓

                    **AI / Human Probability**

                    ↓

                    **Risk Assessment**
                    """
                )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        add_to_history(
            display_name,
            result_label,
            ai_probability,
            risk
        )

        return {
            "Filename": display_name,
            "Result": result_label,
            "AI Probability": f"{ai_probability:.1f}%",
            "Human Probability": f"{human_probability:.1f}%",
            "Risk": risk
        }

    except Exception as e:

        st.error(
            f"❌ Error processing {display_name}: {str(e)}"
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">🎙️</div>

        <div class="hero-title">
            AI Voice <span>Cloning Detector</span>
        </div>

        <div class="hero-subtitle">
            AI voice cloning and impersonation risk detection
        </div>

        <div class="status-line">
            <span class="status-dot"></span>
            MODEL READY FOR ANALYSIS
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# INFO CHIPS
# ============================================================

st.markdown(
    """
    <div class="info-strip">

        <div class="info-chip">
            Machine Learning
        </div>

        <div class="info-chip">
            MFCC Audio Features
        </div>

        <div class="info-chip">
            Voice Forensics
        </div>

        <div class="info-chip">
            Explainable Analysis
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔊 Sample Test",
        "🎤 Record Voice",
        "📤 Upload Audio"
    ]
)

# ============================================================
# SAMPLE TEST
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Test the detection system</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Run the trained model against the provided reference samples.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Test Real Human Voice",
            key="real_sample"
        ):

            if os.path.exists("sample_real.flac"):

                predict_audio(
                    "sample_real.flac",
                    "sample_real.flac",
                    show_details=True
                )

            else:

                st.error(
                    "sample_real.flac not found."
                )

    with col2:

        if st.button(
            "Test AI-Cloned Voice",
            key="fake_sample"
        ):

            if os.path.exists("sample_fake.flac"):

                predict_audio(
                    "sample_fake.flac",
                    "sample_fake.flac",
                    show_details=True
                )

            else:

                st.error(
                    "sample_fake.flac not found."
                )


# ============================================================
# RECORD VOICE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Record and analyze a voice</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Record speech directly from your microphone and run the '
        'detection model after recording.'
        '</div>',
        unsafe_allow_html=True
    )

    mic_input = st.audio_input(
        "Record your voice"
    )

    if mic_input is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(
                mic_input.getbuffer()
            )

            temp_path = tmp.name

        predict_audio(
            temp_path,
            "Microphone Recording",
            show_details=True
        )


# ============================================================
# UPLOAD AUDIO
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">Analyze your own audio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload WAV or FLAC speech recordings for analysis. '
        'Multiple files can be processed together.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=["wav", "flac"],
        accept_multiple_files=True,
        help="Supported formats: WAV and FLAC"
    )

    if uploaded_files:

        # ----------------------------------------------------
        # Single file
        # ----------------------------------------------------

        if len(uploaded_files) == 1:

            uploaded_file = uploaded_files[0]

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

        # ----------------------------------------------------
        # Multiple files
        # ----------------------------------------------------

        else:

            st.markdown(
                '<div class="section-title">'
                'Batch Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            results = []

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

                result = predict_audio(
                    temp_path,
                    uploaded_file.name,
                    show_details=False
                )

                if result is not None:
                    results.append(result)

            if results:

                st.success(
                    f"Successfully analyzed {len(results)} audio files."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown(
        '<div class="section-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Results generated during this session.'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Clear History",
        key="clear_history"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="notice">

        <b>Important:</b>
        Detection results are model-based assessments and should
        not be treated as absolute proof of synthetic speech.
        For high-risk situations, use independent identity
        verification and additional authentication methods.

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI Voice Cloning Detector · Machine-learning based
        audio authenticity analysis
    </div>
    """,
    unsafe_allow_html=True
)
