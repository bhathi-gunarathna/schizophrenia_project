import mne
from pathlib import Path

# --------- PATHS ----------
DATA_PATH = Path("Data/EEG_IBIB")
HEALTHY_PATH = DATA_PATH / "healthy"
SCHZ_PATH = DATA_PATH / "schizophrenia"

# --------- FUNCTION ----------
def preprocess_eeg(edf_file):
    print(f"Processing: {edf_file.name}")

    # Load EDF
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    # Band-pass filter
    raw.filter(l_freq=1.0, h_freq=40.0)

    # Re-reference
    raw.set_eeg_reference("average")

    return raw

# --------- TEST WITH ONE FILE ----------
if __name__ == "__main__":
    test_file = list(HEALTHY_PATH.glob("*.edf"))[0]
    raw = preprocess_eeg(test_file)

    # Plot first 10 seconds
    raw.plot(duration=10)
