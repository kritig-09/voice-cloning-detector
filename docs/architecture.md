# System Architecture

User → Streamlit Frontend (Upload / Live Record / Sample Test) → Audio Preprocessing (librosa) → MFCC Feature Extraction (13 coefficients) → Random Forest Model (voice_cloning_model.pkl) → Prediction: Real Voice / AI-Cloned Voice + Confidence Score → Explainability (Feature Importance) + Spectrogram Visualization
