import kagglehub

# Download latest version
path = kagglehub.dataset_download("rajdalsaniya/pothole-detection-dataset")

print("Path to dataset files:", path)