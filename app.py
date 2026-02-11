from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

from utils.fmri_pipeline import predict_fmri
from utils.eeg_pipeline import predict_eeg

app = FastAPI(
    title="Schizophrenia Screening System (fMRI MVP)",
    description="Upload fMRI (.nii.gz) to detect schizophrenia",
    version="1.0"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict_fmri")
async def predict_fmri_api(file: UploadFile = File(...)):

    # 1. Validate file type
    if not file.filename.endswith(".nii.gz"):
        return {
            "error": "Invalid file format. Please upload a .nii.gz fMRI file"
        }

    # 2. Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Run model inference
    prediction, confidence = predict_fmri(file_path)

    label = "Healthy" if prediction == 0 else "Non-Healthy (Schizophrenia)"

    # 4. Return response
    return {
        "modality": "fMRI",
        "prediction": label,
        "confidence": round(confidence, 3),
        "class_id": prediction,
         "disclaimer": "This result is for research and screening purposes only and is not a medical diagnosis."
    }

@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict_eeg")
async def predict_eeg_api(file: UploadFile = File(...)):
    if not file.filename.endswith(".edf"):
        return {"error": "Please upload a valid .edf EEG file"}

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    prediction, confidence = predict_eeg(file_path)

    return {
        "modality": "EEG",
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "disclaimer": "This is a screening tool, not a medical diagnosis."
    }

    if __name__ == "__main__":
    ...