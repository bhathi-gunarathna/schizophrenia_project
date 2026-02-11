from utils.eeg_pipeline import predict_eeg

edf_file = "Data/EEG_IBIB/healthy/h02.edf"  # change path

label, confidence = predict_eeg(edf_file)

print("Prediction:", label)
print("Confidence:", confidence)
