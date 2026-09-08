# AI Voice Cloning Detector

An AI-powered system for real-time detection of AI-generated (cloned) voices versus authentic human speech. The system uses acoustic feature extraction combined with a supervised machine learning classifier to distinguish genuine human speech from synthetically generated or manipulated audio. Developed as part of Smart India Hackathon 2026.

**Live Application:** https://voice-cloning-detector-sih-2026.streamlit.app/

---

## 1. Project Information

- **Project Title:** AI Voice Cloning Detector
- **PS ID:** SIH26104
- **PS Title:** AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks
- **Organization:** All India Council for Technical Education (AICTE) – Cyber Security Cell
- **Category:** Software
- **Theme:** Blockchain & Cybersecurity / Smart Automation

---

## 2. Problem Statement

Recent advances in generative AI and text-to-speech synthesis have made it possible to clone a person's voice with only a few seconds of reference audio. Tools capable of producing highly convincing synthetic speech are now widely and cheaply available, lowering the barrier for malicious use.

This technology is increasingly being weaponized for:

- **Financial fraud** — impersonating a family member, executive, or bank official to authorize fraudulent transactions ("vishing" and voice-based social engineering).
- **Impersonation of public figures and officials** — fabricating statements attributed to political leaders, celebrities, or corporate executives.
- **Spread of misinformation** — circulating manipulated audio clips that appear authentic, undermining trust in genuine recordings.
- **Bypassing voice-based authentication systems** — used in some banking and call-center verification workflows.

The core challenge is that cloned voices are often acoustically indistinguishable from real speech to an untrained human listener, and no accessible, real-time tool exists for the average user, call center, or platform to verify voice authenticity on demand. This creates an urgent need for an automated, lightweight, and interpretable detection system that can flag synthetic audio before it is used to deceive or defraud.

---

## 3. Proposed Solution

The AI Voice Cloning Detector addresses this gap by providing an accessible web-based tool that classifies any given voice sample as either genuine human speech or AI-generated/cloned audio, along with a confidence score.

**How it works:**

1. **Audio Input** — The user provides an audio sample through one of three channels: uploading a file, recording live via microphone, or selecting a pre-loaded demo sample.
2. **Preprocessing** — The audio is loaded and standardized (resampled to a fixed sampling rate) using `librosa`, and basic quality checks are performed (minimum duration, non-silence, signal strength) to ensure the input is suitable for reliable analysis.
3. **Feature Extraction** — The system computes **Mel-Frequency Cepstral Coefficients (MFCCs)**, a set of 13 coefficients that capture the short-term power spectrum of the audio in a way that closely reflects human auditory perception. MFCCs are a standard and well-validated feature representation in speech processing tasks, including speaker verification and spoof detection.
4. **Classification** — The extracted MFCC feature vector is passed into a **Random Forest classifier**, trained on the ASVspoof 2019 dataset, which outputs a binary prediction (Real / AI-Cloned) along with class probabilities.
5. **Explainability & Visualization** — Alongside the prediction, the system generates a waveform and spectrogram of the input audio, reports audio-level metrics (duration, sample rate, signal quality), and displays a feature-importance chart showing which MFCC coefficients contributed most to the decision — making the system's reasoning transparent rather than a black box.

This approach was chosen over more complex deep-learning pipelines because MFCC + Random Forest offers a strong balance of accuracy, interpretability, low computational cost, and fast inference — making it practical to deploy as a free, publicly accessible web tool rather than requiring heavy GPU infrastructure.

---

## 4. Key Features

- **Multi-file upload** — Upload one or several audio files (`.flac`, `.wav`) at once for batch analysis, with results consolidated into a results table.
- **Live microphone recording** — Record audio directly in the browser and receive an instant prediction, enabling real-time, no-file-needed testing.
- **Built-in demo samples** — Pre-loaded real and AI-cloned audio samples let evaluators test the system instantly without needing their own audio files.
- **Confidence scoring** — Every prediction is accompanied by a probability-based confidence score, visualized with a progress indicator, along with an associated impersonation-risk label.
- **Waveform visualization** — Displays the time-domain waveform of the analyzed audio.
- **Spectrogram visualization** — Displays the time-frequency representation of the analyzed audio for visual inspection.
- **Audio-level metrics** — Reports duration, sample rate, and signal quality for every analyzed sample.
- **Model explainability** — A feature-importance chart shows which MFCC coefficients most influenced a given classification, adding transparency to the model's decision-making.
- **Input validation** — Automatically detects and warns against audio that is too short, silent, or of insufficient quality for reliable prediction, rather than returning a misleading result.
- **Session-based history log** — Maintains a running, clearable table of all predictions made during a session (including confidence, duration, and signal quality), useful for comparing multiple test cases side by side.

---

## 5. Technology Stack

- **Application Framework:** Streamlit (Python-based web app framework, chosen for rapid deployment of ML-driven interfaces)
- **Machine Learning:** Python, scikit-learn (Random Forest Classifier)
- **Audio Processing:** librosa (audio loading, resampling, MFCC extraction, waveform and spectrogram generation)
- **Visualization:** Matplotlib
- **Dataset:** ASVspoof 2019 (Logical Access) — an internationally recognized benchmark dataset for spoofed and synthetic speech detection, containing both bonafide (real) and spoofed (AI-generated/replayed) utterances
- **Deployment:** Streamlit Community Cloud

---

## 6. Architecture

Detailed architecture available in [docs/architecture.md](docs/architecture.md).

```
User
  |
  v
Streamlit Frontend (Sample Test / Live Record / Upload Audio)
  |
  v
Audio Preprocessing + Validation (librosa)
  |
  v
MFCC Feature Extraction (13 coefficients)
  |
  v
Random Forest Model (voice_cloning_model.pkl)
  |
  v
Prediction: Real Voice / AI-Cloned Voice + Confidence Score
  |
  v
Waveform + Spectrogram Visualization + MFCC Explainability + Session History
```

---

## 7. Repository Structure

```
voice-cloning-detector/
├── README.md
├── SUBMISSION_GUIDE.md
├── submission/
│   ├── PRESENTATION.md
│   └── DEMO.md
├── src/
│   └── app.py
├── docs/
│   └── architecture.md
├── assets/
│   └── screenshots/
├── requirements.txt
├── packages.txt
├── .gitignore
└── LICENSE
```

### What goes where?

| Item | Location |
|---|---|
| Source code | `src/` |
| Architecture / technical documentation | `docs/` |
| Application screenshots | `assets/screenshots/` |
| Final presentation | `submission/PRESENTATION.md` |
| Demo video link | `submission/DEMO.md` |
| Project overview | `README.md` |

---

## 8. Final Presentation

See [submission/PRESENTATION.md](submission/PRESENTATION.md) for the presentation link.

---

## 9. Demo Video

See [submission/DEMO.md](submission/DEMO.md) for the demo video link.

---

## 10. Screenshots

Application screenshots are available in [assets/screenshots/](assets/screenshots/).

---

## 11. Installation

```bash
git clone https://github.com/kritig-09/voice-cloning-detector.git
cd voice-cloning-detector
pip install -r requirements.txt
```

---

## 12. Run

```bash
streamlit run src/app.py
```

---

## 13. Future Scope

- **Speaker verification module** — extending the system to check whether a specific individual's voice is being used without their consent, via speaker embedding and similarity matching.
- **Broader format support** — adding native support for additional audio formats such as MP3 and M4A.
- **Deep learning-based classification** — exploring CNN/RNN architectures operating directly on spectrograms to improve robustness against unseen, more sophisticated cloning techniques.
- **Real-time call integration** — integrating the detection pipeline into live communication platforms (VoIP, call centers) for real-time fraud alerts during phone calls.
- **Mobile application** — building a lightweight mobile version for on-device voice verification without requiring an internet connection.

---

## Important

This repository does not contain any passwords, API keys, access tokens, or confidential credentials.
