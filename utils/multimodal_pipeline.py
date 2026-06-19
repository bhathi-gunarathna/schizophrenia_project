from utils.eeg_pipeline import predict_eeg
from utils.fmri_pipeline import predict_fmri


def late_fusion_predict(eeg_path, fmri_path):

    eeg_pred, eeg_conf = predict_eeg(eeg_path)

    fmri_pred, fmri_conf = predict_fmri(fmri_path)

    # Convert confidence to schizophrenia probability

    eeg_prob = eeg_conf if eeg_pred == 1 else (1 - eeg_conf)

    fmri_prob = fmri_conf if fmri_pred == 1 else (1 - fmri_conf)

    print("EEG Schizophrenia Probability:", eeg_prob)
    print("fMRI Schizophrenia Probability:", fmri_prob)

    # Weighted fusion

    fusion_prob = (
        eeg_prob * 0.6 +
        fmri_prob * 0.4
    )

    if fusion_prob >= 0.5:

        label = "Non-Healthy (Schizophrenia)"

    else:

        label = "Healthy"

    return label, round(fusion_prob ,3)