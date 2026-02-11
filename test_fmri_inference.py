from utils.fmri_pipeline import predict_fmri

file_path = "Data/fMRI_preprocessed/healthy/sub-01_task-speech_bold_cleaned.nii.gz"

pred, prob = predict_fmri(file_path)

label = "Healthy" if pred == 0 else "Non-Healthy"

print("Prediction:", label)
print("Probability:", round(prob, 3))
