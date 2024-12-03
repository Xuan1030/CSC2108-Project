import torch
import clip
from PIL import Image
from tuneNewClassifier import ImageDataset
import os
import random

def image_inference(image, text):
    # Get image and text embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize the embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return image_features, text_features
    
if __name__ == "__main__":
    # Set device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the CLIP model and preprocess
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    unique_countries = []

    for folder in os.listdir("./compressed_dataset"):
        if (folder != ".DS_Store"):
            unique_countries.append(f"A street view of {folder}")

    images = []
    countries = []


    for foldername, subfolders, filenames in os.walk("./compressed_dataset"):
        if foldername == "./compressed_dataset":
            continue
        country = os.path.basename(foldername)  # Get the folder name (country)
        
        for filename in filenames:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(foldername, filename)
            images.append(img_path)
            countries.append(f"A street view of {country}")  # Add the integer label
            
    # print(len(images))
    # print(len(countries))
    
    # Test the first 300 images from the dataset
    # Combine the lists into a single list of tuples
    combined = list(zip(images, countries))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unpack the shuffled list back into two lists
    images_shuffled, countries_shuffled = zip(*combined)

    # Convert back to lists (optional, if you need list objects)
    images_shuffled = list(images_shuffled)
    countries_shuffled = list(countries_shuffled)
    
    test_data_length = int(len(images_shuffled) * 0.2)
    print(test_data_length)
    images = images_shuffled[:300]
    countries = countries_shuffled[:300]
    
    correct_predictions = 0
    total_images = len(images)
    top5_correct_predictions = 0
    for i in range(total_images):
        # Load and preprocess an image
        image_path = images[i] # Replace with your image path
        actual_country = countries[i]
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        # Prepare text inputs
        text = clip.tokenize(unique_countries).to(device)

        image_features, text_features = image_inference(image, text)
        
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).squeeze(0)  # Shape: (num_text_prompts,)
        # print("Similarity scores:", similarity)

        # Identify the best-matching text description
        best_match_idx = similarity.argmax().item()

        print(f"Best match: {best_match_idx}, Text: {unique_countries[best_match_idx]}")
        if (unique_countries[best_match_idx] == actual_country):
            correct_predictions += 1
        
        top_5_values, top_5_indices = torch.topk(similarity, k=5)
        best_5_preds = [unique_countries[idx] for idx in top_5_indices.tolist()]
        print(f"Top 5 matches: {best_5_preds}")
        if (actual_country in best_5_preds):
            top5_correct_predictions += 1


    print(f"Top-1 Accuracy: {correct_predictions / total_images}")
    print(f"Top-5 Accuracy: {top5_correct_predictions / total_images}")