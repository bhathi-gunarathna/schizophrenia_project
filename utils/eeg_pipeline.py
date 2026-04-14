import mne
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/eeg_model.pkl")
scaler = joblib.load("models/eeg_scaler.pkl")


def extract_eeg_features(file_path):

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    raw.pick_types(eeg=True)

    if len(raw.ch_names) >= 16:
        raw.pick(raw.ch_names[:16])
    else:
        raise ValueError("Not enough EEG channels")

    raw.filter(1, 40)

    data = raw.get_data()

    mean = np.mean(data, axis=1)
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

    # Ensure fixed size (95)
    expected_len = scaler.n_features_in_

    if len(features) > expected_len:
        features = features[:expected_len]
    elif len(features) < expected_len:
        features = np.pad(features, (0, expected_len - len(features)))

    return features


def predict_eeg(file_path):

    features = extract_eeg_features(file_path)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    pred = model.predict(features)[0]
    score = model.decision_function(features)[0]

    prob = 1 / (1 + np.exp(-score))   # sigmoid

    return int(pred), float(prob)