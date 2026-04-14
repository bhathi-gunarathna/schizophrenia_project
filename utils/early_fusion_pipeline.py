import numpy as np
import joblib

from utils.eeg_pipeline import extract_eeg_features
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

        # Ensure EEG feature size
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

        # Ensure correct input size for PCA
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

        # Scale
        fused = scaler.transform(fused)

        # ---------------- PREDICTION ----------------
        prediction = model.predict(fused)[0]

        # ---------------- CONFIDENCE (FIXED) ----------------
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(fused)[0][prediction]
        else:
            score = model.decision_function(fused)[0]
            confidence = 1 / (1 + np.exp(-score))

        return int(prediction), float(confidence)

    except Exception as e:
        print("Fusion Error:", e)
        return 0, 0.0