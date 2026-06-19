import mne
import numpy as np
from pathlib import Path

# ---------------- PATHS ----------------
DATA_PATH = Path("Data/EEG_IBIB")
HEALTHY_PATH = DATA_PATH / "healthy"
SCHZ_PATH = DATA_PATH / "schizophrenia"

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_eeg(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

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
        "gamma": (30, 40)
    }

    psd = raw.compute_psd(
        method="welch",
        fmin=1,
        fmax=40,
        n_fft=1024,
        verbose=False
    )

    psds = psd.get_data()       # shape: (channels, frequencies)
    freqs = psd.freqs

    features = []

    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        band_power = psds[:, idx].mean(axis=1)  # mean over freq bins
        features.append(band_power)

    features = np.concatenate(features)  # shape: (bands × channels,)
    return features


# ---------------- MAIN LOOP ----------------
X = []
y = []

# Healthy controls (label = 0)
for edf_file in HEALTHY_PATH.glob("*.edf"):
    print(f"Processing healthy: {edf_file.name}")
    raw = preprocess_eeg(edf_file)
    features = extract_bandpower_features(raw)

    X.append(features)
    y.append(0)

# Schizophrenia patients (label = 1)
for edf_file in SCHZ_PATH.glob("*.edf"):
    print(f"Processing schizophrenia: {edf_file.name}")
    raw = preprocess_eeg(edf_file)
    features = extract_bandpower_features(raw)

    X.append(features)
    y.append(1)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Save features

output_dir = Path("Data/features")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "X_ibib.npy", X)
np.save(output_dir / "y_ibib.npy", y)

print("✅ Feature extraction completed and saved.")
