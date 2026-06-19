import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Data
X = np.load("Data/features/X_aszed.npy")
y = np.load("Data/features/y_aszed.npy")

print("Dataset shape:", X.shape)


# 80 / 10 / 10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
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


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# Normalize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Hyperparameter tuning
C_values = [0.01, 0.1, 1, 10, 100]

best_acc = 0
best_model = None
best_C = None

for C in C_values:

    model = SVC(kernel="linear", C=C)

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    acc = accuracy_score(y_val, val_pred)

    print(f"C={C} -> Val Accuracy={acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_C = C


print("\nBest C:", best_C)


# Final test
test_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))


print("\n ASZED EEG Model Training Complete")