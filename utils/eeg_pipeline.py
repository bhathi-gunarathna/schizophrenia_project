import mne
import numpy as np
import joblib

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("models/eeg_aszed_model.pkl")
scaler = joblib.load("models/eeg_aszed_scaler.pkl")


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_eeg_features(file_path):

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=False
    )

    # EEG channels only
    raw.pick_types(eeg=True)

    # Same as training
    eeg_channels = raw.ch_names[:19]

    raw.pick_channels(eeg_channels)

    # Same as training
    raw.set_eeg_reference("average")

    # Same as training
    raw.filter(1, 40)

    data = raw.get_data()

    # Safety check
    expected_channels = 19

    if data.shape[0] < expected_channels:

        pad = expected_channels - data.shape[0]

        data = np.pad(
            data,
            ((0, pad), (0, 0)),
            mode="constant"
        )

    elif data.shape[0] > expected_channels:

        data = data[:expected_channels]

    # ---------------- FEATURES ----------------

    mean = np.mean(data, axis=1)

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

    if np.isnan(features).any():
        raise ValueError("NaN detected in EEG features")

    return features


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_eeg(file_path):

    try:

        features = extract_eeg_features(file_path)

        print("Feature Length:", len(features))

        features = features.reshape(1, -1)

        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        confidence = probabilities[prediction]

        print("Probabilities:", probabilities)
        print("Prediction:", prediction)

        return int(prediction), float(confidence)

    except Exception as e:

        print("EEG Error:", e)

        return 0, 0.0