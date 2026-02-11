# Data/models/train_fmri.py

import os
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# =====================================================
# 1. LOAD FEATURES
# =====================================================

X = np.load("Data/features/fmri_X.npy")
y = np.load("Data/features/fmri_y.npy")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# =====================================================
# 2. CREATE MODELS DIRECTORY (FIX)
# =====================================================

os.makedirs("models", exist_ok=True)

# =====================================================
# 3. SCALING + PCA
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA output shape: {X_pca.shape}")

# =====================================================
# 4. MODEL DEFINITION
# =====================================================

model = SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced",
    random_state=42
)

# =====================================================
# 5. CROSS-VALIDATION (REPORTING)
# =====================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
acc_scores = []

fold = 1
for train_idx, test_idx in cv.split(X_pca, y):

    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    acc_scores.append(acc)
    auc_scores.append(auc)

    print(f"Fold {fold} → AUC: {auc:.4f} | ACC: {acc:.4f}")
    fold += 1

print("\n===== FINAL RESULTS =====")
print(f"Mean AUC: {np.mean(auc_scores):.4f}")
print(f"Mean ACC: {np.mean(acc_scores):.4f}")

# =====================================================
# 6. FINAL TRAINING ON FULL DATA
# =====================================================

model.fit(X_pca, y)

# =====================================================
# 7. SAVE EVERYTHING (FIXED)
# =====================================================

joblib.dump(model, "models/fmri_model.pkl")
joblib.dump(scaler, "models/fmri_scaler.pkl")
joblib.dump(pca, "models/fmri_pca.pkl")

print("\n✅ fMRI model, scaler, and PCA saved successfully")
