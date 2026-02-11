import numpy as np
import joblib
import mne

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "models/eeg_model.pkl"
SCALER_PATH = "models/eeg_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- PREPROCESS ----------------
def preprocess_eeg(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    raw.filter(1.0, 40.0)
    raw.set_eeg_reference("average")

    return raw

# ---------------- FEATURE EXTRACTION ----------------
def extract_bandpower_features(raw):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    psd = raw.compute_psd(
        method="welch",
        fmin=1,
        fmax=40,
        n_fft=1024,
        verbose=False
    )

    psds = psd.get_data()
    freqs = psd.freqs

    features = []

    for fmin, fmax in bands.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        band_power = psds[:, idx].mean(axis=1)
        features.append(band_power)

    return np.concatenate(features)

# ---------------- PREDICTION ----------------
def predict_eeg(edf_path):
    raw = preprocess_eeg(edf_path)
    features = extract_bandpower_features(raw)

    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prob = model.predict_proba(features)[0][1]
    pred = int(prob >= 0.5)

    label = "Schizophrenia" if pred == 1 else "Healthy"
    confidence = prob if pred == 1 else 1 - prob

    return label, round(float(confidence), 3)
