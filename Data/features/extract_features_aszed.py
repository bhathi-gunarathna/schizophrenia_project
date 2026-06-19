import os
import numpy as np
import mne

DATA_DIR = "Data/EEG_ASZED"
FEATURE_DIR = "Data/features"

os.makedirs(FEATURE_DIR, exist_ok=True)


def extract_features(raw):

    # Pick EEG channels only
    raw.pick_types(eeg=True)

    # Limit to 16 channels
    if len(raw.ch_names) >= 16:
        raw.pick(raw.ch_names[:16])
    else:
        return None  # skip small channel files

    # Bandpass filter
    raw.filter(1, 40)

    data = raw.get_data()

    # Mean
    mean = np.mean(data, axis=1)

    # Std
    std = np.std(data, axis=1)

    sfreq = raw.info['sfreq']

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1,
        fmax=40
    )

    delta = np.mean(psd[:, (freqs >= 1) & (freqs <= 4)], axis=1)
    theta = np.mean(psd[:, (freqs >= 4) & (freqs <= 8)], axis=1)
    alpha = np.mean(psd[:, (freqs >= 8) & (freqs <= 13)], axis=1)
    beta  = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)], axis=1)

    features = np.concatenate([mean, std, delta, theta, alpha, beta])

    return features


X = []
y = []

# Healthy
healthy_path = os.path.join(DATA_DIR, "healthy")

for file in os.listdir(healthy_path):

    if file.endswith(".edf"):

        print("Processing healthy:", file)

        try:
            raw = mne.io.read_raw_edf(
                os.path.join(healthy_path, file),
                preload=True,
                verbose=False
            )

            features = extract_features(raw)

            if features is not None and len(features) == 96:
                X.append(features)
                y.append(0)

        except Exception as e:
            print("Skipped:", file)


# Schizophrenia
schz_path = os.path.join(DATA_DIR, "schizophrenia")

for file in os.listdir(schz_path):

    if file.endswith(".edf"):

        print("Processing schizophrenia:", file)

        try:
            raw = mne.io.read_raw_edf(
                os.path.join(schz_path, file),
                preload=True,
                verbose=False
            )

            features = extract_features(raw)

            if features is not None and len(features) == 96:
                X.append(features)
                y.append(1)

        except Exception as e:
            print("Skipped:", file)


X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("Data/features/X_aszed.npy", X)
np.save("Data/features/y_aszed.npy", y)

print("ASZED Feature Extraction Complete")