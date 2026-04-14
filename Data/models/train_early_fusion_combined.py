import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib


# ---------------- LOAD DATA ----------------
X_eeg = np.load("Data/features/X_eeg_combined.npy")
y_eeg = np.load("Data/features/y_eeg_combined.npy")

X_fmri = np.load("Data/features/fmri_X.npy")
y_fmri = np.load("Data/features/fmri_y.npy")

print("EEG shape:", X_eeg.shape)
print("fMRI shape:", X_fmri.shape)

# ---------------- NORMALIZE EEG ----------------
eeg_scaler = StandardScaler()
X_eeg = eeg_scaler.fit_transform(X_eeg)

# ---------------- REDUCE fMRI DIMENSION ----------------
print("Original fMRI shape:", X_fmri.shape)

pca = PCA(n_components=50)   # 🔥 reduce 1128 → 50
X_fmri = pca.fit_transform(X_fmri)

print("Reduced fMRI shape:", X_fmri.shape)

# ---------------- NORMALIZE fMRI ----------------
fmri_scaler = StandardScaler()
X_fmri = fmri_scaler.fit_transform(X_fmri)


# ---------------- ALIGN DATA ----------------
n = min(len(X_eeg), len(X_fmri))

X_eeg = X_eeg[:n]
y = y_eeg[:n]

X_fmri = X_fmri[:n]

print("Using samples:", n)

# ---------------- FUSION ----------------
X_fused = np.concatenate([X_eeg, X_fmri], axis=1)

print("Fused shape:", X_fused.shape)


# ---------------- SPLIT ----------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_fused, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# ---------------- NORMALIZATION ----------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
C_values = [0.01, 0.1, 1, 10, 100]

best_acc = 0
best_model = None
best_C = None

for C in C_values:
    model = SVC(kernel="linear", C=C)

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)

    print(f"C={C} -> Val Acc={acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_C = C

print("\nBest C:", best_C)

# ---------------- TEST ----------------
test_pred = best_model.predict(X_test)

print("\nEarly Fusion Accuracy:", accuracy_score(y_test, test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

print("\n Early Fusion (Combined EEG + fMRI) Complete")


from sklearn.model_selection import cross_val_score

model = SVC(kernel="linear", C=0.1, class_weight="balanced")

scores = cross_val_score(model, X_fused, y, cv=5)

print("\nCross-validation accuracy:", scores)
print("Mean accuracy:", scores.mean())


# Save model
joblib.dump(best_model, "models/early_fusion_model.pkl")

# Save PCA
joblib.dump(pca, "models/pca.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model, PCA, Scaler saved successfully")