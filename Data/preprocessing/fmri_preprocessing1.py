import os
from nilearn.image import clean_img
from multiprocessing import Pool, cpu_count

RAW_DIR = "Data/fMRI"
OUT_DIR = "Data/fMRI_preprocessed"

def process_file(args):
    in_path, out_path = args
    
    cleaned = clean_img(
        in_path,
        detrend=True,
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0
    )
    
    cleaned.to_filename(out_path)
    print(f"Saved: {out_path}")

def collect_files():
    tasks = []

    for group in ["healthy", "schizophrenia"]:
        in_dir = os.path.join(RAW_DIR, group)
        out_dir = os.path.join(OUT_DIR, group)
        os.makedirs(out_dir, exist_ok=True)

        for file in os.listdir(in_dir):
            if file.endswith(".nii.gz"):
                in_path = os.path.join(in_dir, file)
                out_path = os.path.join(
                    out_dir,
                    file.replace(".nii.gz", "_cleaned.nii.gz")
                )
                tasks.append((in_path, out_path))
    
    return tasks

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    
    tasks = collect_files()

    n_cores = cpu_count() - 1
    print(f"Using {n_cores} CPU cores...")

    with Pool(n_cores) as p:
        p.map(process_file, tasks)
