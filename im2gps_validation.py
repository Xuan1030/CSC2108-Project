import torch, clip
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tuneNewClassifier import ImageDataset, extract_image_embeddings

def find_image_index(dataset, query_image_tensor):
    for idx, (image, _) in enumerate(dataset):  # Iterate through dataset
        if torch.equal(image, query_image_tensor):  # Compare tensors
            return idx
    return -1  # Return -1 if no match is found

def idx_to_country(idx):
    return next(key for key, value in dataset.country_to_index.items() if value == idx)

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
    dataset = ImageDataset(dataset_folder="./datasets/img2country3ktest", transform=transform)

    torch.manual_seed(42)  # Set seed for reproducibility
    test_dataset = dataset
    
    train_embed_path = "./validations/train_image_features.pt"
    test_embed_path = "./validations/test_image_features.pt"
    # Create DataLoaders for Training and Testing
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    '''training/testing images and labels'''
    test_image_features, test_labels = extract_image_embeddings(test_dataloader, model, device)

    # Save the embeddings for reuse
    torch.save(test_image_features, "./validations/test_image_features.pt")
    torch.save(test_labels, "./validations/test_labels.pt")

    print("Features extracted and saved for im2gpt validation.")
    
    # Load the extracted features 
    test_image_features = torch.load("./validations/test_image_features.pt", weights_only=True)
    # Country labels
    train_labels = torch.load("./validations/country_labels.pt", weights_only=True)
    
    test_image_features = test_image_features.to(device)
    train_labels = train_labels.to(device)

    num_classes = len(torch.unique(train_labels))  # Total number of unique countries
    
    classifier = nn.Linear(512, num_classes).to(device)
    state_dict = torch.load("./training_new_classifier/classifier.pth", weights_only=True)  # Load the saved state dict
    classifier.load_state_dict(state_dict) 

    # Set the classifier to evaluation mode
    classifier.eval()

    # Get the test image features
    with torch.no_grad():
        test_image_features = test_image_features.to(device)
        test_outputs = classifier(test_image_features)
        preds = torch.argmax(test_outputs, dim=1)
        top1_accuracy = (preds == test_labels).float().mean().item()

        # Top-5 predictions
        top5_preds = torch.topk(test_outputs, k=5, dim=1).indices  # Shape: (num_samples, 5)
        top5_accuracy = (test_labels.unsqueeze(1) == top5_preds).any(dim=1).float().mean().item()

    print(f"Top-1 Accuracy: {top1_accuracy * 100:.1f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.1f}%")
    