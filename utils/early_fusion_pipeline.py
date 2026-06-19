import numpy as np
import joblib

from utils.eeg_pipeline import extract_eeg_features, predict_eeg
from utils.fmri_pipeline import extract_fmri_features

# --------------------------------------------------
# Load trained objects (ONCE)
# --------------------------------------------------

model = joblib.load("models/early_fusion_model.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")


# --------------------------------------------------
# EARLY FUSION PREDICTION
# --------------------------------------------------

def early_fusion_predict(eeg_path, fmri_path):

    try:
        # ---------------- EEG ----------------
        eeg_features = extract_eeg_features(eeg_path)
        eeg_features = np.array(eeg_features).reshape(1, -1)

        # Normalize EEG (IMPORTANT FIX)
        eeg_features = eeg_features / (np.max(np.abs(eeg_features)) + 1e-6)

        # Ensure EEG size
        expected_eeg_len = 95
        if eeg_features.shape[1] != expected_eeg_len:
            if eeg_features.shape[1] > expected_eeg_len:
                eeg_features = eeg_features[:, :expected_eeg_len]
            else:
                pad = expected_eeg_len - eeg_features.shape[1]
                eeg_features = np.pad(eeg_features, ((0, 0), (0, pad)))

        # ---------------- fMRI ----------------
        fmri_features = extract_fmri_features(fmri_path)
        fmri_features = np.array(fmri_features).reshape(1, -1)

        # Ensure PCA input size
        expected_fmri_len = pca.n_features_in_

        if fmri_features.shape[1] != expected_fmri_len:
            if fmri_features.shape[1] > expected_fmri_len:
                fmri_features = fmri_features[:, :expected_fmri_len]
            else:
                pad = expected_fmri_len - fmri_features.shape[1]
                fmri_features = np.pad(fmri_features, ((0, 0), (0, pad)))

        # Apply PCA
        fmri_features = pca.transform(fmri_features)

        # ---------------- FUSION ----------------
        fused = np.concatenate([eeg_features, fmri_features], axis=1)

        # Scale (trained scaler)
        fused = scaler.transform(fused)

        # ---------------- PROBABILITY ----------------
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(fused)[0][1]
        else:
            score = model.decision_function(fused)[0]
            prob = 1 / (1 + np.exp(-score))

        # ---------------- THRESHOLD TUNING ----------------
        # (IMPORTANT: improves schizophrenia detection)
        prediction = 1 if prob > 0.4 else 0

        # ---------------- FALLBACK (VERY IMPORTANT) ----------------
        # If fusion is uncertain → trust EEG more
        eeg_pred, eeg_conf = predict_eeg(eeg_path)

        if prob < 0.55:
            prediction = eeg_pred
            prob = eeg_conf

        confidence = prob * 100

        return int(prediction), float(confidence)

    except Exception as e:
        print("Fusion Error:", e)
        return 0, 0.0