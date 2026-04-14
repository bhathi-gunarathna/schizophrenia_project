import os
import shutil
import pandas as pd

# ---------- PATHS ----------
DATASET_PATH = "Data/ASZED"
CSV_PATH = "Data/ASZED/ASZED_SpreadSheet.csv"

OUTPUT_PATH = "Data/EEG_ASZED"
HEALTHY_PATH = os.path.join(OUTPUT_PATH, "healthy")
SCHZ_PATH = os.path.join(OUTPUT_PATH, "schizophrenia")

# ---------- CREATE FOLDERS ----------
os.makedirs(HEALTHY_PATH, exist_ok=True)
os.makedirs(SCHZ_PATH, exist_ok=True)

# ---------- LOAD CSV ----------
df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns)
print(df.head())

# ---------- LOOP SUBSETS ----------
for subset in ["subset_1", "subset_2", "subset_3"]:

    subset_path = os.path.join(DATASET_PATH, subset)

    if not os.path.exists(subset_path):
        continue

    print(f"Processing {subset}")

    for subject in os.listdir(subset_path):

        subject_path = os.path.join(subset_path, subject)

        if not os.path.isdir(subject_path):
            continue

        # Find subject in CSV
        subject_row = df[df["sn"] == subject]

        if subject_row.empty:
            print(f"Not found in CSV: {subject}")
            continue

        diagnosis = subject_row.iloc[0]["category"]

        # Decide output folder
        if diagnosis.lower() == "control":
            dest_folder = HEALTHY_PATH
        else:
            dest_folder = SCHZ_PATH

        # Copy EDF files
        for root, dirs, files in os.walk(subject_path):
            for file in files:
                if file.endswith(".edf"):

                    src = os.path.join(root, file)

                    new_name = f"{subject}_{file}"

                    dst = os.path.join(dest_folder, new_name)

                    shutil.copy(src, dst)

print("ASZED Dataset Split Completed")