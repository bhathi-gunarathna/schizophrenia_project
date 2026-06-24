import os
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. LOAD FEATURES
# =====================================================

X = np.load("Data/features/fmri_X.npy")
y = np.load("Data/features/fmri_y.npy")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Create models directory
os.makedirs("models", exist_ok=True)

# =====================================================
# 2. CROSS-VALIDATION (NO DATA LEAKAGE)
# =====================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
precision_scores = []
recall_scores = []
auc_scores = []
f1_scores = []

fold = 1

for train_idx, test_idx in cv.split(X, y):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- Scaling (fit ONLY on training)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- PCA (fit ONLY on training)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # ---- Model
    model = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    y_prob = model.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_prob)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    acc_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    auc_scores.append(auc)
    f1_scores.append(f1)

    print(
        f"Fold {fold} → "
        f"ACC: {acc:.4f} | "
        f"PREC: {precision:.4f} | "
        f"REC: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"AUC: {auc:.4f}"
    )

    fold += 1

print("\n===== FINAL RESULTS =====")
print(f"Mean ACC       : {np.mean(acc_scores):.4f}")
print(f"Mean Precision : {np.mean(precision_scores):.4f}")
print(f"Mean Recall    : {np.mean(recall_scores):.4f}")
print(f"Mean F1        : {np.mean(f1_scores):.4f}")
print(f"Mean AUC       : {np.mean(auc_scores):.4f}")

# Scale full dataset
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

# PCA on full dataset
final_pca = PCA(n_components=0.95, random_state=42)
X_pca = final_pca.fit_transform(X_scaled)

print(f"PCA output shape: {X_pca.shape}")

# Final model
final_model = SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced",
    random_state=42
)

final_model.fit(X_pca, y)
y_pred = final_model.predict(X_pca)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y, y_pred))

# =====================================================
# 4. SAVE MODEL + SCALER + PCA
# =====================================================

joblib.dump(final_model, "models/fmri_model.pkl")
joblib.dump(final_scaler, "models/fmri_scaler.pkl")
joblib.dump(final_pca, "models/fmri_pca.pkl")

print("\n✅ fMRI model, scaler, and PCA saved successfully")
