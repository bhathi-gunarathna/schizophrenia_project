import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ---------------- LOAD DATA ----------------

X = np.load("Data/features/X_aszed.npy")
y = np.load("Data/features/y_aszed.npy")

print("Dataset Shape:", X.shape)

print("Healthy:", np.sum(y == 0))
print("Schizophrenia:", np.sum(y == 1))

# ---------------- SPLIT 80 / 10 / 10 ----------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ---------------- SCALE ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---------------- HYPERPARAMETER TUNING ----------------

C_values = [0.01, 0.1, 1, 10, 100]

best_model = None
best_acc = 0
best_C = None

for C in C_values:

    model = SVC(
        kernel="linear",
        C=C,
        class_weight="balanced",
        probability=True
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    acc = accuracy_score(y_val, val_pred)

    print(f"C={C} -> Validation Accuracy={acc:.3f}")

    if acc > best_acc:

        best_acc = acc
        best_model = model
        best_C = C

print("\nBest C:", best_C)

# ---------------- TEST ----------------

y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

# ---------------- SAVE ----------------

joblib.dump(
    best_model,
    "models/eeg_aszed_model.pkl"
)

joblib.dump(
    scaler,
    "models/eeg_aszed_scaler.pkl"
)

print("\n✅ EEG ASZED Model Saved")