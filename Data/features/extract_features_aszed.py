import os
import numpy as np
import mne

OUTPUT_DIR = "Data/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Feature Extraction
# ---------------------------

def extract_features(raw):

    raw.pick_types(eeg=True)

    eeg_channels = raw.ch_names[:19]

    raw.pick_channels(eeg_channels)

    raw.set_eeg_reference("average")

    raw.filter(1, 40)

    data = raw.get_data()

    # Mean
    mean = np.mean(data, axis=1)

    # Std
    std = np.std(data, axis=1)

    sfreq = raw.info["sfreq"]

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1,
        fmax=40
    )

    delta = np.mean(
        psd[:, (freqs >= 1) & (freqs <= 4)],
        axis=1
    )

    theta = np.mean(
        psd[:, (freqs >= 4) & (freqs <= 8)],
        axis=1
    )

    alpha = np.mean(
        psd[:, (freqs >= 8) & (freqs <= 13)],
        axis=1
    )

    beta = np.mean(
        psd[:, (freqs >= 13) & (freqs <= 30)],
        axis=1
    )

    gamma = np.mean(
        psd[:, (freqs >= 30) & (freqs <= 40)],
        axis=1
    )

    features = np.concatenate([
        mean,
        std,
        delta,
        theta,
        alpha,
        beta,
        gamma
    ])

    return features


# ---------------------------
# Build Dataset
# ---------------------------

X = []
y = []

for label, folder in [

    (0, "Data/EEG_ASZED/healthy"),
    (1, "Data/EEG_ASZED/schizophrenia")

]:

    for file in os.listdir(folder):

        if not file.endswith(".edf"):
            continue

        try:

            raw = mne.io.read_raw_edf(
                os.path.join(folder, file),
                preload=True,
                verbose=False
            )

            features = extract_features(raw)

            X.append(features)
            y.append(label)

            print("Processed:", file)

        except Exception as e:

            print("Skipped:", file)
            print(e)

X = np.array(X)
y = np.array(y)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

np.save(
    "Data/features/X_aszed.npy",
    X
)

np.save(
    "Data/features/y_aszed.npy",
    y
)

print("Dataset Saved Successfully")