from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

from utils.fmri_pipeline import predict_fmri
from utils.eeg_pipeline import predict_eeg
from utils.multimodal_pipeline import multimodal_predict

app = FastAPI(
    title="SchzoFusion – Schizophrenia Screening System",
    description="Multimodal AI platform for schizophrenia screening using fMRI and EEG.",
    version="1.0"
)

# Serve static files (CSS, JS assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.get("/fmri")
def fmri_page():
    return FileResponse("static/fmri.html")

@app.get("/eeg")
def eeg_page():
    return FileResponse("static/eeg.html")

@app.get("/multimodal")
def multimodal_page():
    return FileResponse("static/multimodal.html")

@app.get("/about")
def about_page():
    return FileResponse("static/about.html")


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

@app.post("/predict_multimodal")
async def predict_multimodal(
        eeg_file: UploadFile = File(None),
        fmri_file: UploadFile = File(None)):

    os.makedirs("temp", exist_ok=True)

    eeg_path = None
    fmri_path = None

    if eeg_file:
        eeg_path = f"temp/{eeg_file.filename}"
        with open(eeg_path, "wb") as f:
            f.write(await eeg_file.read())

    if fmri_file:
        fmri_path = f"temp/{fmri_file.filename}"
        with open(fmri_path, "wb") as f:
            f.write(await fmri_file.read())

    label, confidence = multimodal_predict(eeg_path, fmri_path)

    return {
        "prediction": label,
        "confidence": confidence,
        "class_id": 1 if label == "Suffering from Schizophrenia" else 0
    }
