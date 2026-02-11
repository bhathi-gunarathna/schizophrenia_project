import os
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# ==============================
# PATHS
# ==============================

DATA_DIR = "Data/fMRI_preprocessed"
OUTPUT_DIR = "features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD AAL ATLAS
# ==============================

print("Loading AAL atlas...")

aal = datasets.fetch_atlas_aal()
atlas_img = aal.maps

masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

# ==============================
# FEATURE EXTRACTION
# ==============================

X = []
y = []

def process_group(group_name, label):
    group_path = os.path.join(DATA_DIR, group_name)

    print(f"\nProcessing group: {group_name}")

    for file in os.listdir(group_path):
        if file.endswith("_cleaned.nii.gz"):
            file_path = os.path.join(group_path, file)
            print(f"Processing: {file}")

            # Extract ROI time series
            time_series = masker.fit_transform(file_path)

            # Compute correlation matrix
            corr_matrix = np.corrcoef(time_series.T)

            # Take upper triangle (exclude diagonal)
            iu = np.triu_indices_from(corr_matrix, k=1)
            features = corr_matrix[iu]

            X.append(features)
            y.append(label)

# Healthy = 0
process_group("healthy", 0)

# Schizophrenia = 1
process_group("schizophrenia", 1)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# ==============================
# SAVE OUTPUT
# ==============================

np.save(os.path.join(OUTPUT_DIR, "X_fmri.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_fmri.npy"), y)

print("\nFeature extraction complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
