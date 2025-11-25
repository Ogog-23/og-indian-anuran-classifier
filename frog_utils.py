# frog_utils.py

from pathlib import Path
import json

import numpy as np
import librosa
import scipy.signal as sps
import joblib

# === constants (same as notebook) ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64

ROOT = Path(r"D:\Frog_Calls_Raven")  # same ROOT as your notebook

# === load scaler and label mapping ===
SCALER_PATH = "clean_aux_scaler.joblib"
LABELS_PATH = "label_mapping.json"

scaler = joblib.load(SCALER_PATH)

# label_mapping.json is a simple list:
# ["Raven_amboli", "Raven_coorg", "Raven_knob"]
with open(LABELS_PATH, "r") as f:
    species_order = json.load(f)

label2idx = {lab: i for i, lab in enumerate(species_order)}
idx2label = {i: lab for lab, i in label2idx.items()}
n_classes = len(species_order)

# AUX_COLS from notebook
AUX_COLS = ["peak_log", "active_s"]


# ========== basic signal helpers (same as notebook) ==========

def bandpass_filter(y, sr=SR, low=200.0, high=6000.0, order=4):
    nyq = sr / 2.0
    lown, highn = low / nyq, high / nyq
    b, a = sps.butter(order, [lown, highn], btype='band')
    try:
        return sps.filtfilt(b, a, y)
    except Exception:
        return y


def pre_emphasis(y, coef=0.97):
    return np.append(y[0], y[1:] - coef * y[:-1])


def spectral_gate(y, sr=SR, n_fft=512, hop_length=256, prop_decrease=0.9):
    """Simple spectral gating noise reduction."""
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_profile = np.median(mag, axis=1, keepdims=True)
    mag_denoised = mag - prop_decrease * noise_profile
    mag_denoised = np.maximum(mag_denoised, 1e-10)
    S = mag_denoised * np.exp(1j * phase)
    y_out = librosa.istft(S, hop_length=hop_length, length=len(y))
    return y_out


# NOTE: set this to the TARGET_FRAMES printed in the notebook.
# For your current trained model it was 372; keep that unless you retrain.
TARGET_FRAMES = 372


def preprocess_waveform_for_spec(y, sr=SR):
    """
    EXACTLY the same preprocessing used for training in frog_clean_pipeline.ipynb.
    """
    y = bandpass_filter(y, sr, low=200.0, high=6000.0)
    y = pre_emphasis(y)
    y = spectral_gate(y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, prop_decrease=0.9)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return y.astype(np.float32)


def make_mel_spec_for_array(y):
    """
    Takes 1D waveform array y (sr=SR),
    applies same preprocessing as training and returns
    a mel-spectrogram [TARGET_FRAMES, N_MELS, 1].
    """
    y = preprocess_waveform_for_spec(y, SR)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    t = S_db.shape[1]
    if t < TARGET_FRAMES:
        S_db = np.pad(
            S_db,
            ((0, 0), (TARGET_FRAMES - t, 0)),
            mode="constant",
            constant_values=S_db.min()
        )
    else:
        # take the last TARGET_FRAMES columns, exactly like notebook
        S_db = S_db[:, -TARGET_FRAMES:]

    spec = np.expand_dims(S_db.T.astype(np.float32), -1)
    return spec


def compute_aux_features_for_array(y):
    """
    Compute peak_log + active_s and scale with the fitted scaler.
    Uses the same logic as training.
    """
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.95

    # active duration
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    med = np.median(rms) if rms.size > 0 else 0.0
    thresh = max(1e-8, 0.35 * med)
    active_frames = (rms >= thresh).sum()
    active_s = active_frames * 256.0 / SR

    # spectral centroid → peak-ish frequency
    try:
        cent = librosa.feature.spectral_centroid(
            y=y, sr=SR, n_fft=512, hop_length=256
        )[0]
        peak_hz = float(np.median(cent)) if cent.size > 0 else 0.0
    except Exception:
        peak_hz = 0.0

    peak_log = np.log1p(peak_hz)

    # shape (1, 2) for scaler
    aux_arr = np.array([[peak_log, active_s]], dtype=np.float32)
    aux_scaled = scaler.transform(aux_arr)[0].astype(np.float32)  # shape (2,)

    return aux_scaled
