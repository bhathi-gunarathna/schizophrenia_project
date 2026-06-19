from utils.fmri_pipeline import predict_fmri
from utils.eeg_pipeline import predict_eeg


def multimodal_predict(eeg_file=None, fmri_file=None):

    eeg_prob = None
    fmri_prob = None

    # ---------------- EEG prediction ----------------
    if eeg_file:
        eeg_label, eeg_prob = predict_eeg(eeg_file)

    # ---------------- fMRI prediction ----------------
    if fmri_file:
        fmri_label, fmri_prob = predict_fmri(fmri_file)

    # ---------------- Fusion ----------------
    probs = []

    if eeg_prob is not None:
        probs.append(eeg_prob)

    if fmri_prob is not None:
        probs.append(fmri_prob)

    if len(probs) == 0:
        return "No data provided", 0

    # weighted fusion (fMRI slightly stronger)
    if eeg_prob is not None and fmri_prob is not None:
        final_prob = 0.4 * eeg_prob + 0.6 * fmri_prob
    else:
        final_prob = probs[0]

    # ---------------- Final decision ----------------
    if final_prob >= 0.5:
        label = "Suffering from Schizophrenia"
        class_id = 1
    else:
        label = "Healthy"
        class_id = 0

    return label, final_prob