import numpy as np

# Load both datasets
X_ibib = np.load("Data/features/X_ibib.npy")
y_ibib = np.load("Data/features/y_ibib.npy")

X_aszed = np.load("Data/features/X_aszed.npy")
y_aszed = np.load("Data/features/y_aszed.npy")

print("IBIB:", X_ibib.shape)
print("ASZED:", X_aszed.shape)

# ⚠️ Feature size mismatch fix
min_features = min(X_ibib.shape[1], X_aszed.shape[1])

X_ibib = X_ibib[:, :min_features]
X_aszed = X_aszed[:, :min_features]

# Combine
X_combined = np.vstack((X_ibib, X_aszed))
y_combined = np.concatenate((y_ibib, y_aszed))

print("Combined:", X_combined.shape)

# Save
np.save("Data/features/X_eeg_combined.npy", X_combined)
np.save("Data/features/y_eeg_combined.npy", y_combined)

print("EEG Combined Dataset Created")