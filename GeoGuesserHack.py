import os
import clip
import torch
from torchvision import transforms
from PIL import Image, ImageGrab
import ssl
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import time

def test_single_image(image_path, model, classifier, device, idx_to_country_func):
    # Image preprocessing pipeline
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)
    
    # Extract CLIP feature
    with torch.no_grad():
        image_feature = model.encode_image(processed_image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

    # Pass the feature through the classifier
    classifier.eval()
    with torch.no_grad():
        logits = classifier(image_feature)
        probs = F.softmax(logits, dim=1)

    # Get top 5 predictions
    top_k = 5
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
    top_probs = top_probs.squeeze(0).tolist() 
    top_indices = top_indices.squeeze(0).tolist() 

    # Print the results
    print(f"Predictions for Image: {image_path}")
    for i in range(top_k):
        country_name = idx_to_country_func(top_indices[i] + 1)
        print(f"Class: {country_name}, Probability: {top_probs[i]:.2%}")

def idx_to_country(idx):
    country_to_index = {country: idx for idx, country in enumerate(sorted(os.listdir("./compressed_dataset"))) if '.' not in country}
    return next(key for key, value in country_to_index.items() if value == idx)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
classifier = nn.Linear(512, 124).to(device)
state_dict = torch.load("./training_new_classifier/classifier.pth", weights_only=True)  # Load the saved state dict
classifier.load_state_dict(state_dict) 

def process_image(image):
    # Save the image for reference
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("/Users/xuan/Desktop/GeoGuesser_Hack", f"screenshot_{timestamp}.png")
    image.save(image_path)

    # Call your classifier here
    print("Running classifier on the image...")
    test_single_image(image_path, model, classifier, device, idx_to_country)

while True:
    try:
        # Grab image from clipboard
        image = ImageGrab.grabclipboard()
        if isinstance(image, Image.Image):
            print("Image detected in clipboard.")
            process_image(image)
    except Exception as e:
        print(f"Error grabbing image: {e}")
