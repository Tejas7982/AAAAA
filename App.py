import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib

model1 = joblib.load("C:/Users/Tejas/OneDrive/Desktop/modelForPrediction1.sav")
def process_audio(file):
    # Load the audio file
    signal, sr = librosa.load(file, sr=44100)

    # Extract features
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(signal, sr=sr)
    mel = librosa.feature.melspectrogram(signal, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(signal, sr=sr)

    # Concatenate features
    features = np.concatenate([mfccs, chroma, mel, spectral_contrast], axis=0)

    # Normalize features
    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    features = (features - mean[:, np.newaxis]) / std[:, np.newaxis]

    return features

st.title("Audio Classification App")

# Upload audio file
audio_file = st.file_uploader("Upload audio file", type=["mp3"])

if audio_file is not None:
    # Process audio file
    features = process_audio(audio_file)

    # Make predictions
    pred1 = model1.predict(features)

    # Display predictions
    st.write("Model 1 prediction:", pred1[0])

