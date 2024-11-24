import kagglehub

# Download latest version
path = kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")

print("Path to dataset files:", path)
