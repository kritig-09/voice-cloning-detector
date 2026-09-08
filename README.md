# AI Voice Cloning Detector

AI-powered system to detect whether a given audio sample is a real human voice or an AI-cloned (spoofed) voice, using MFCC audio features and a Random Forest classifier. Built for Smart India Hackathon 2026.

**Live App:** https://voice-cloning-detector-sih2026.streamlit.app/

## 1. Project Information

- **Project Title:** AI Voice Cloning Detector
- **PS ID:** *(add your problem statement ID)*
- **PS Title:** *(add your problem statement title)*
- **Category:** Software
- **Theme:** *(add your SIH theme)*

## 2. Problem Statement

AI voice cloning technology has made it increasingly easy to generate synthetic voices that closely mimic real human speech. This poses serious risks — including fraud, impersonation, misinformation, and unauthorized use of a person's voice — since it is often difficult for an average listener to distinguish a real voice from an AI-generated one.

## 3. Proposed Solution

This project allows a user to upload or record an audio sample. The system extracts MFCC (Mel-Frequency Cepstral Coefficients) features from the audio and passes them through a trained Random Forest classifier, which predicts whether the voice is real or AI-cloned, along with a confidence score.

## 4. Key Features

- Upload audio files (.flac, .wav) — single or multiple at once
- Real-time microphone recording and detection
- Built-in sample audio for instant demo (real + AI-cloned)
- Confidence score with visual progress indicator
- Spectrogram visualization of the analyzed audio
- Model explainability — shows which MFCC features influenced the decision
- Audio quality/duration validation with error handling
- Prediction history within the session

## 5. Technology Stack

- **Frontend/App:** Streamlit
- **Backend/ML:** Python, scikit-learn (Random Forest), librosa
- **Audio Processing:** librosa, MFCC feature extraction
- **Visualization:** Matplotlib
- **Dataset:** ASVspoof 2019 (Logical Access)
- **Deployment:** Streamlit Community Cloud

## 6. Architecture

See [docs/architecture.md](docs/architecture.md).
