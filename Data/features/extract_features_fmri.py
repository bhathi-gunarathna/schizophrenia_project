import os
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img
import warnings


os.makedirs("Data/features", exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = "Data/fMRI_preprocessed"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Harvard-Oxford atlas...")

atlas = datasets.fetch_atlas_harvard_oxford(
    atlas_name="cort-maxprob-thr25-2mm"
)

# -----------------------------
# Reference image
# -----------------------------
first_group = os.listdir(DATA_DIR)[0]
first_file = os.listdir(os.path.join(DATA_DIR, first_group))[0]
reference_img = os.path.join(DATA_DIR, first_group, first_file)

print("Using reference image:", reference_img)

resampled_atlas = resample_to_img(
    atlas.maps,
    reference_img,
    interpolation="nearest"
)

masker = NiftiLabelsMasker(
    labels_img=resampled_atlas,
    standardize="zscore_sample",
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

X_list = []
y = []
feature_lengths = []

# -----------------------------
# Feature extraction
# -----------------------------
def process_group(group_name, label):
    group_path = os.path.join(DATA_DIR, group_name)

    print(f"\nProcessing group: {group_name}")

    for file in sorted(os.listdir(group_path)):
        if file.endswith("_cleaned.nii.gz"):
            file_path = os.path.join(group_path, file)
            print("Processing:", file)

            ts = masker.fit_transform(file_path)
            corr = np.corrcoef(ts.T)

            iu = np.triu_indices_from(corr, k=1)
            features = corr[iu]

            X_list.append(features)
            y.append(label)
            feature_lengths.append(len(features))

process_group("healthy", 0)
process_group("schizophrenia", 1)

# -----------------------------
# Zero-padding (CRITICAL FIX)
# -----------------------------
max_len = max(feature_lengths)
print("Max feature length:", max_len)

X = np.zeros((len(X_list), max_len))

for i, feat in enumerate(X_list):
    X[i, :len(feat)] = feat

y = np.array(y)

# -----------------------------
# Save
# -----------------------------
np.save(os.path.join(OUTPUT_DIR, "X_fmri.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_fmri.npy"), y)

print("\n✅ Feature extraction complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("Data/features/fmri_X.npy", X)
np.save("Data/features/fmri_y.npy", y)

print("💾 fMRI features saved to disk")