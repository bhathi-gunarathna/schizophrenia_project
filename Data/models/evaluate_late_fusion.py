import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Example predictions
eeg_probs = np.load("Data/features/eeg_probs.npy")
fmri_probs = np.load("Data/features/fmri_probs.npy")
y_true = np.load("Data/features/fusion_labels.npy")

fusion_probs = (
    eeg_probs * 0.6 +
    fmri_probs * 0.4
)

y_pred = (fusion_probs >= 0.5).astype(int)

print("Accuracy:",
      accuracy_score(y_true, y_pred))

print("Precision:",
      precision_score(y_true, y_pred))

print("Recall:",
      recall_score(y_true, y_pred))

print("F1:",
      f1_score(y_true, y_pred))