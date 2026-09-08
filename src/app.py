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
    layout="wide"
)

# ============================================================
# MATPLOTLIB
# ============================================================

matplotlib.rcParams.update({
    "figure.facecolor": "#0e1015",
    "axes.facecolor": "#0e1015",
    "axes.edgecolor": "#30343d",
    "axes.labelcolor": "#c7ccd6",
    "text.color": "#e7e9ee",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af",
})

# ============================================================
# CSS
# ============================================================

st.markdown(
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Manrope", sans-serif;
}

.stApp {
    background: #0e1015;
    color: #e7e9ee;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ---------------- HERO ---------------- */

.hero {
    text-align: center;
    padding: 25px 10px 20px 10px;
}

.hero-icon {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #f8fafc;
    margin: 0;
}

.hero-accent {
    color: #818cf8;
}

.hero-subtitle {
    margin-top: 10px;
    color: #9ba3b2;
    font-size: 1rem;
    font-weight: 500;
}

/* ---------------- STATUS ---------------- */

.status {
    display: inline-block;
    margin-top: 17px;
    padding: 6px 14px;
    border: 1px solid #29303b;
    border-radius: 999px;
    background: #13161d;
    color: #aeb6c5;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
}

/* ---------------- INFO ---------------- */

.info-text {
    text-align: center;
    color: #777f8e;
    font-size: 0.8rem;
    margin: 8px 0 28px 0;
}

/* ---------------- SECTION ---------------- */

.section-title {
    font-size: 1.3rem;
    font-weight: 750;
    color: #f1f3f6;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-description {
    color: #858e9d;
    font-size: 0.87rem;
    margin-bottom: 18px;
}

/* ---------------- TABS ---------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    border-bottom: 1px solid #242932;
}

.stTabs [data-baseweb="tab"] {
    color: #8f97a6;
    font-weight: 650;
    font-size: 0.9rem;
    padding: 11px 18px;
}

.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
}

/* ---------------- BUTTONS ---------------- */

.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 10px;
    border: 1px solid #343b49;
    background: #171a22;
    color: #e8ebf0;
    font-family: "Manrope", sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
}

.stButton > button:hover {
    border-color: #6366f1;
    background: #1d2130;
    color: #ffffff;
}

/* ---------------- UPLOADER ---------------- */

div[data-testid="stFileUploader"] {
    background: #12151c;
    border: 1px dashed #39414e;
    border-radius: 12px;
    padding: 10px;
}

/* ---------------- RESULT ---------------- */

.result-box {
    text-align: center;
    background: #12151c;
    border: 1px solid #303641;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
}

.result-heading {
    color: #858e9d;
    font-size: 0.72rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 7px;
}

.result-sub {
    color: #929baa;
    font-size: 0.85rem;
    margin-top: 5px;
}

/* ---------------- SMALL CARDS ---------------- */

.card {
    background: #12151c;
    border: 1px solid #282e38;
    border-radius: 12px;
    padding: 18px;
}

.card-label {
    color: #7f8795;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.card-value {
    color: #edf0f4;
    font-size: 1.25rem;
    font-weight: 750;
    margin-top: 4px;
}

/* ---------------- RISK ---------------- */

.risk {
    background: #12151c;
    border: 1px solid #303641;
    border-radius: 12px;
    padding: 15px 18px;
    margin: 15px 0;
}

.risk-title {
    font-size: 0.95rem;
    font-weight: 800;
}

.risk-description {
    color: #8f98a7;
    font-size: 0.8rem;
    margin-top: 5px;
}

/* ---------------- NOTICE ---------------- */

.notice {
    background: #11141a;
    border: 1px solid #2c333e;
    border-radius: 11px;
    padding: 14px 17px;
    color: #9da5b4;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 15px;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    color: #626a78;
    font-size: 0.72rem;
    margin-top: 50px;
    padding-top: 18px;
    border-top: 1px solid #20242c;
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
except Exception:
    st.error("Could not load voice_cloning_model.pkl")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# AUDIO QUALITY
# ============================================================

def audio_quality(audio, sr):

    duration = len(audio) / sr

    peak = float(np.max(np.abs(audio))) if len(audio) else 0

    rms = float(
        np.sqrt(np.mean(audio ** 2))
    ) if len(audio) else 0

    if peak < 0.01:
        level = "Very Low"
    elif peak < 0.08:
        level = "Low"
    elif peak < 0.8:
        level = "Good"
    else:
        level = "High"

    if rms < 0.005:
        signal = "Very Low"
    elif rms < 0.02:
        signal = "Low"
    else:
        signal = "Good"

    return {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "level": level,
        "signal": signal
    }


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk(ai_probability):

    if ai_probability >= 70:

        return (
            "HIGH RISK",
            "Strong indication of AI-generated or cloned speech.",
            "high"
        )

    elif ai_probability >= 30:

        return (
            "SUSPICIOUS",
            "Some characteristics associated with synthetic speech were detected.",
            "medium"
        )

    else:

        return (
            "LOW RISK",
            "The signal shows stronger characteristics associated with human speech.",
            "low"
        )


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(audio, sr, title):

    fig, ax = plt.subplots(figsize=(9, 3.3))

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=11,
        pad=10
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# MFCC IMPORTANCE
# ============================================================

def show_feature_importance():

    if not hasattr(
        model,
        "feature_importances_"
    ):

        st.info(
            "Feature importance is not available "
            "for this model type."
        )

        return

    importances = np.asarray(
        model.feature_importances_
    )

    names = [
        f"MFCC-{i + 1}"
        for i in range(len(importances))
    ]

    order = np.argsort(
        importances
    )[::-1]

    names = [
        names[i]
        for i in order
    ]

    values = importances[order]

    fig, ax = plt.subplots(
        figsize=(9, 3.4)
    )

    ax.bar(
        names,
        values
    )

    ax.set_title(
        "MFCC Feature Influence",
        fontsize=11
    )

    ax.set_ylabel(
        "Relative Importance"
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=8
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.caption(
        "Most influential features: "
        + ", ".join(names[:3])
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_result(
    result,
    ai_probability,
    human_probability,
    risk,
    risk_description
):

    if result == "AI-Cloned Voice":

        icon = "⚠️"

    else:

        icon = "✓"

    st.markdown(
        f"""
<div class="result-box">
<div class="result-heading">
Voice Authenticity Assessment
</div>
<div class="result-value">
{icon} {result}
</div>
<div class="result-sub">
Model confidence: {max(ai_probability, human_probability):.1f}%
</div>
</div>
""",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
<div class="card">
<div class="card-label">
AI-Generated Probability
</div>
<div class="card-value">
{ai_probability:.1f}%
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
<div class="card">
<div class="card-label">
Human Voice Probability
</div>
<div class="card-value">
{human_probability:.1f}%
</div>
</div>
""",
            unsafe_allow_html=True
        )

    if risk == "HIGH RISK":

        risk_color = "#fb7185"

    elif risk == "SUSPICIOUS":

        risk_color = "#fbbf24"

    else:

        risk_color = "#34d399"

    st.markdown(
        f"""
<div class="risk">
<div class="risk-title" style="color:{risk_color};">
{risk}
</div>
<div class="risk-description">
{risk_description}
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.progress(
        float(ai_probability / 100),
        text=f"AI probability: {ai_probability:.1f}%"
    )

    if ai_probability >= 70:

        st.markdown(
            """
<div class="notice">
<b>Security recommendation</b><br>
Do not rely on voice alone for high-risk decisions.
Verify the speaker using an independent communication
channel or an additional authentication factor.
</div>
""",
            unsafe_allow_html=True
        )

    elif ai_probability >= 30:

        st.markdown(
            """
<div class="notice">
<b>Verification recommended</b><br>
This result is not conclusive. Consider additional
authentication before trusting the recording.
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# FORENSICS
# ============================================================

def show_forensics(audio, sr, quality):

    st.markdown(
        '<div class="section-title">Audio Forensics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Technical characteristics extracted from the analyzed signal.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
<div class="card">
<div class="card-label">Duration</div>
<div class="card-value">{quality["duration"]:.1f}s</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
<div class="card">
<div class="card-label">Sample Rate</div>
<div class="card-value">{sr // 1000} kHz</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
<div class="card">
<div class="card-label">Channels</div>
<div class="card-value">Mono</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
<div class="card">
<div class="card-label">Features</div>
<div class="card-value">13 MFCC</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
<div class="notice">
<b>Signal level:</b> {quality["level"]}<br>
<b>Signal quality:</b> {quality["signal"]}<br>
<b>Peak amplitude:</b> {quality["peak"]:.4f}<br>
<b>RMS energy:</b> {quality["rms"]:.4f}
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# PREDICT
# ============================================================

def predict_audio(
    path,
    display_name,
    show_details=True
):

    try:

        audio, sr = librosa.load(
            path,
            sr=16000,
            mono=True
        )

        quality = audio_quality(
            audio,
            sr
        )

        duration = quality["duration"]

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if duration < 0.5:

            st.warning(
                "Audio is too short. Please provide "
                "at least 0.5 seconds of speech."
            )

            return None

        if quality["peak"] < 0.01:

            st.warning(
                "Audio appears silent or has extremely "
                "low volume."
            )

            return None

        # ----------------------------------------------------
        # AUDIO PLAYER
        # ----------------------------------------------------

        if show_details:

            st.audio(
                path
            )

            st.caption(
                f"Analyzing: {display_name}"
            )

        # ----------------------------------------------------
        # MFCC
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
        # MODEL
        # ----------------------------------------------------

        prediction = model.predict(
            mfcc_mean
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                mfcc_mean
            )[0]

            try:

                classes = list(
                    model.classes_
                )

                ai_index = classes.index(0)
                human_index = classes.index(1)

                ai_probability = (
                    probabilities[ai_index] * 100
                )

                human_probability = (
                    probabilities[human_index] * 100
                )

            except Exception:

                ai_probability = (
                    probabilities[0] * 100
                )

                human_probability = (
                    probabilities[1] * 100
                )

        else:

            if prediction == 1:

                human_probability = 100
                ai_probability = 0

            else:

                human_probability = 0
                ai_probability = 100

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if prediction == 1:

            result = "Real Voice"

        else:

            result = "AI-Cloned Voice"

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk, risk_description, risk_type = get_risk(
            ai_probability
        )

        # ----------------------------------------------------
        # SHOW DETAILS
        # ----------------------------------------------------

        if show_details:

            display_result(
                result,
                ai_probability,
                human_probability,
                risk,
                risk_description
            )

            show_forensics(
                audio,
                sr,
                quality
            )

            # ------------------------------------------------
            # SPECTROGRAM
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                'Spectral Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Frequency-domain representation of the speech signal.'
                '</div>',
                unsafe_allow_html=True
            )

            show_spectrogram(
                audio,
                sr,
                f"Speech Spectrogram — {display_name}"
            )

            # ------------------------------------------------
            # EXPLAINABILITY
            # ------------------------------------------------

            with st.expander(
                "🧠 Why this decision? — Model Explainability"
            ):

                st.write(
                    "The model represents speech using "
                    "13 Mel-Frequency Cepstral Coefficients "
                    "(MFCCs). These features capture important "
                    "spectral characteristics of the voice."
                )

                show_feature_importance()

            # ------------------------------------------------
            # SECURITY
            # ------------------------------------------------

            with st.expander(
                "🛡️ Recommended Security Response"
            ):

                st.write(
                    "For sensitive situations, voice detection "
                    "should be combined with independent identity "
                    "verification."
                )

                st.write(
                    "Recommended actions:"
                )

                st.write(
                    "• Verify the speaker through a trusted channel."
                )

                st.write(
                    "• Use an additional authentication factor."
                )

                st.write(
                    "• Avoid sharing sensitive information until "
                    "identity is confirmed."
                )

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            with st.expander(
                "⚙️ How the Detection Works"
            ):

                st.write(
                    "Audio Input"
                )

                st.write(
                    "↓"
                )

                st.write(
                    "Audio Normalization → 16 kHz"
                )

                st.write(
                    "↓"
                )

                st.write(
                    "MFCC Feature Extraction → 13 Features"
                )

                st.write(
                    "↓"
                )

                st.write(
                    "Machine-Learning Classifier"
                )

                st.write(
                    "↓"
                )

                st.write(
                    "AI / Human Probability"
                )

                st.write(
                    "↓"
                )

                st.write(
                    "Risk Assessment"
                )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history_entry = {
            "Filename": display_name,
            "Result": result,
            "AI Probability": f"{ai_probability:.1f}%",
            "Risk": risk
        }

        if (
            not st.session_state.history
            or st.session_state.history[-1] != history_entry
        ):

            st.session_state.history.append(
                history_entry
            )

        return history_entry

    except Exception as e:

        st.error(
            f"Error processing {display_name}: {str(e)}"
        )

        return None


# ============================================================
# HERO
# ============================================================

st.markdown(
"""
<div class="hero">
<div class="hero-icon">🎙️</div>
<div class="hero-title">
AI Voice <span class="hero-accent">Cloning Detector</span>
</div>
<div class="hero-subtitle">
AI voice cloning and impersonation risk detection
</div>
<div class="status">
● MODEL READY FOR ANALYSIS
</div>
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="info-text">
Machine Learning &nbsp;•&nbsp;
MFCC Audio Features &nbsp;•&nbsp;
Voice Forensics &nbsp;•&nbsp;
Explainable Analysis
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
        "🎤 Record Voice",
        "📤 Upload Audio"
    ]
)

# ============================================================
# SAMPLE
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        'Test the detection system'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Use the provided reference recordings to demonstrate '
        'the model.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Test Real Human Voice",
            key="real"
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

    with c2:

        if st.button(
            "Test AI-Cloned Voice",
            key="fake"
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
# RECORD
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        'Record and analyze a voice'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Record speech directly from your microphone and '
        'analyze the recording with the trained model.'
        '</div>',
        unsafe_allow_html=True
    )

    mic = st.audio_input(
        "Record your voice"
    )

    if mic is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp:

            temp.write(
                mic.getbuffer()
            )

            temp_path = temp.name

        predict_audio(
            temp_path,
            "Microphone Recording",
            show_details=True
        )


# ============================================================
# UPLOAD
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        'Analyze your own audio'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload WAV or FLAC speech recordings for analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose audio files",
        type=["wav", "flac"],
        accept_multiple_files=True
    )

    if uploaded:

        # ----------------------------------------------------
        # SINGLE
        # ----------------------------------------------------

        if len(uploaded) == 1:

            file = uploaded[0]

            suffix = os.path.splitext(
                file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp:

                temp.write(
                    file.getbuffer()
                )

                temp_path = temp.name

            predict_audio(
                temp_path,
                file.name,
                show_details=True
            )

        # ----------------------------------------------------
        # MULTIPLE
        # ----------------------------------------------------

        else:

            st.markdown(
                '<div class="section-title">'
                'Batch Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            results = []

            for file in uploaded:

                suffix = os.path.splitext(
                    file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp:

                    temp.write(
                        file.getbuffer()
                    )

                    temp_path = temp.name

                result = predict_audio(
                    temp_path,
                    file.name,
                    show_details=False
                )

                if result:

                    results.append(
                        result
                    )

            if results:

                st.success(
                    f"Analyzed {len(results)} audio files."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.markdown(
        '<div class="section-title">'
        'Prediction History'
        '</div>',
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
        "Clear History"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
"""
<div class="notice">
<b>Important:</b>
Detection results are model-based assessments and should not
be treated as absolute proof of synthetic speech. For high-risk
situations, use independent identity verification and additional
authentication methods.
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
AI Voice Cloning Detector · Machine-learning based audio authenticity analysis
</div>
""",
unsafe_allow_html=True
)
