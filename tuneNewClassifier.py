'''Used in tuneNewClassifier.py'''
import os
import clip
import torch
from torchvision import transforms
from PIL import Image
import ssl
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
print({country: idx for idx, country in enumerate(sorted(os.listdir("./compressed_dataset"))) if '.' not in country}["United States"])
# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def plot_image(img_path):
    """Plot an image given its file path."""
    # Open the image using PIL
    img = Image.open(img_path)
    
    # Plot the image using matplotlib
    plt.imshow(img)
    plt.axis('off')  # Turn off axes for better visualization
    plt.title(f"Image: {img_path.split('/')[-1]}")  # Add image name as title
    plt.show()
    
# Define the Dataset
class ImageDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.images = []
        self.countries = []
        self.transform = transform

        # Create a mapping of folder names (countries) to integer indices
        self.country_to_index = {country: idx for idx, country in enumerate(sorted(os.listdir(dataset_folder))) if '.' not in country}

        for foldername, subfolders, filenames in os.walk(dataset_folder):
            if foldername == "./compressed_dataset":
                continue
            country = os.path.basename(foldername)  # Get the folder name (country)
            if country not in self.country_to_index:
                continue  # Skip if the folder is not in the mapping
            
            for filename in filenames:
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(foldername, filename)
                self.images.append(img_path)
                self.countries.append(self.country_to_index[country])  # Add the integer label
                
    def __len__(self):
        return len(self.countries)
    
    def __getitem__(self, idx):
        # print(f"Loading item {idx}")
        image_path = self.images[idx]
        country_label = self.countries[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, country_label
    
# Feature extraction function
def extract_image_embeddings(dataloader, model, device):
    image_features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:  # Assuming dataloader provides images and integer labels
            images = images.to(device)

            # Encode image embeddings
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize features

            image_features_list.append(image_features)
            labels_list.append(labels)

    # Combine all batches into single tensors
    image_features = torch.cat(image_features_list, dim=0)  # Shape: [num_samples, 512]
    labels = torch.cat(labels_list, dim=0)  # Shape: [num_samples]

    return image_features, labels

def find_image_index(dataset, query_image_tensor):
    for idx, (image, _) in enumerate(dataset):  # Iterate through dataset
        if torch.equal(image, query_image_tensor):  # Compare tensors
            return idx
    return -1  # Return -1 if no match is found

def idx_to_country(idx):
    return next(key for key, value in dataset.country_to_index.items() if value == idx)


# Ensure multiprocessing safety
if __name__ == "__main__":
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Initialize Dataset and DataLoader
    dataset = ImageDataset(dataset_folder="compressed_dataset", transform=transform)
    img, text = dataset[0]
    # print(f"Image: {img}, text: {text}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)  # Set seed for reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders for Training and Testing
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    '''  Save training/testing images and labels
    # Example usage:
    train_image_features, train_labels = extract_image_embeddings(train_dataloader, model, device)
    test_image_features, test_labels = extract_image_embeddings(test_dataloader, model, device)

    # Save the embeddings for reuse
    torch.save(train_image_features, "train_image_features.pt")
    torch.save(train_labels, "train_labels.pt")
    torch.save(test_image_features, "test_image_features.pt")
    torch.save(test_labels, "test_labels.pt")

    print("Features extracted and saved for training and testing sets.")
    '''
    
    # Load the extracted features 
    train_image_features = torch.load("./training_new_classifier/train_image_features.pt", weights_only=True)
    test_image_features = torch.load("./training_new_classifier/test_image_features.pt", weights_only=True)
    # Country labels
    train_labels = torch.load("./training_new_classifier/train_labels.pt", weights_only=True)
    test_labels = torch.load("./training_new_classifier/test_labels.pt", weights_only=True)
    
    train_image_features = train_image_features.to(device)
    test_image_features = test_image_features.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    num_classes = len(torch.unique(train_labels))  # Total number of unique countries
    
    classifier = nn.Linear(512, num_classes).to(device)
    state_dict = torch.load("./training_new_classifier/classifier.pth", weights_only=True)  # Load the saved state dict
    classifier.load_state_dict(state_dict) 
    ''' Model Saved at ./training_new_classifier!!!
    
    classifier = nn.Linear(512, num_classes).to(device)
    # Define the Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)


    epochs = 100
    batch_size = 64

    # Split train features into batches
    num_train_samples = train_image_features.shape[0]
    train_dataset = torch.utils.data.TensorDataset(train_image_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        correct_predictions = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()

        train_accuracy = correct_predictions / num_train_samples
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%")
    '''
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(test_image_features)
        # Top-1 Predictions Accuracy
        preds = torch.argmax(outputs, dim=1)
        top1_accuracy = (preds == test_labels).float().mean().item()

        # Top-5 predictions
        top5_preds = torch.topk(outputs, k=5, dim=1).indices  # Shape: (num_samples, 5)
        top5_accuracy = (test_labels.unsqueeze(1) == top5_preds).any(dim=1).float().mean().item()

    print(f"Top-1 Accuracy: {top1_accuracy * 100:.1f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.1f}%")
    
    ''' ALREADY SAVED
    torch.save(classifier.state_dict(), "./training_new_classifier/classifier.pth")
    print("Classifier saved successfully.")
    '''
    
    print()
    # Test with a random selected image
    random_idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[random_idx]
    img_path_index = find_image_index(dataset, image_tensor)
    print()
    print(f"True country: {idx_to_country(label)}")
    # print("RANDOM INDEX:", random_idx, "img_path_index:", img_path_index)
    plot_image(dataset.images[img_path_index])
    # Predict using the trained classifier
    classifier.eval()
    with torch.no_grad():
        # Retrieve the precomputed CLIP feature for the test image
        image_feature = test_image_features[random_idx]
        
        # Unsqueeze to add batch dimension (if needed for the classifier)
        image_feature = image_feature.unsqueeze(0).to(device)

        # Pass the feature to the classifier
        logits = classifier(image_feature)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        # print(probs.shape)
        # Get top 5 predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
        top_probs = top_probs.squeeze(0).tolist()  # Remove batch dim
        top_indices = top_indices.squeeze(0).tolist()  # Remove batch dim

    # Print the results
    print("Top 5 Predictions:")
    for i in range(top_k):
        print(f"Class: {idx_to_country(top_indices[i] + 1)}, Probability: {top_probs[i]}")
        