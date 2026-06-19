import joblib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

# --------------------------------------------------
# Load trained objects (ONCE)
# --------------------------------------------------

model = joblib.load("models/fmri_model.pkl")
scaler = joblib.load("models/fmri_scaler.pkl")
pca = joblib.load("models/fmri_pca.pkl")

# --------------------------------------------------
# Load atlas
# --------------------------------------------------

atlas = datasets.fetch_atlas_harvard_oxford(
    "cort-maxprob-thr25-2mm"
)

masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize="zscore_sample"
)

# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_fmri(file_path: str):

    # 1. Extract ROI time-series
    time_series = masker.fit_transform(file_path)

    # 2. Functional connectivity
    corr = np.corrcoef(time_series.T)
    features = corr[np.triu_indices_from(corr, k=1)]

    # --------------------------------------------------
    # 🔥 CRITICAL FIX: feature length alignment
    # --------------------------------------------------
    expected_len = scaler.n_features_in_

    if features.shape[0] < expected_len:
        # Pad with zeros
        pad_width = expected_len - features.shape[0]
        features = np.pad(features, (0, pad_width))

    elif features.shape[0] > expected_len:
        # Truncate
        features = features[:expected_len]

    # Reshape for sklearn
    features = features.reshape(1, -1)

    # 3. Apply scaler + PCA
    features = scaler.transform(features)
    features = pca.transform(features)

    # 4. Predict
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0, 1]

    return int(pred), float(prob)