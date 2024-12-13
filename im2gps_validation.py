import torch, clip, os
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


def find_image_index(dataset, query_image_tensor):
    for idx, (image, _) in enumerate(dataset):  # Iterate through dataset
        if torch.equal(image, query_image_tensor):  # Compare tensors
            return idx
    return -1  # Return -1 if no match is found

def idx_to_country(dataset, idx):
    return next(key for key, value in dataset.country_to_index.items() if value == idx)



class ImageDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.images = []
        self.countries = []
        self.country_labels = []
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
                self.country_labels.append(self.country_to_index[country])  # Add the integer label
                self.countries.append(country)
                
    def __len__(self):
        return len(self.countries)
    
    def __getitem__(self, idx):
        # print(f"Loading item {idx}")
        image_path = self.images[idx]
        country_label = self.country_labels[idx]
        country = self.countries[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, country_label, country


def extract_image_embeddings(dataloader, model, device):
    image_features_list = []
    labels_list = []
    countries_list = []

    with torch.no_grad():
        for images, labels, countries in dataloader:  # Assuming dataloader provides images and integer labels
            images = images.to(device)

            # Encode image embeddings
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize features

            image_features_list.append(image_features)
            labels_list.append(labels)
            countries_list.append(countries[0])

    # Combine all batches into single tensors
    image_features = torch.cat(image_features_list, dim=0)  # Shape: [num_samples, 512]
    labels = torch.cat(labels_list, dim=0)  # Shape: [num_samples]

    return image_features, labels, countries_list


def idx_to_country(country_2_idx, idx):
    return next(key for key, value in country_2_idx.items() if value == idx)


if __name__ == "__main__":
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for inference.")
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

    country_2_idx = ImageDataset(dataset_folder="./datasets/compressed_dataset", transform=transform).country_to_index
    
    # Initialize Dataset and DataLoader
    test_dataset = ImageDataset(dataset_folder="./datasets/img2country3ktest", transform=transform)

    torch.manual_seed(42)  # Set seed for reproducibility

    # # Create DataLoaders for Training and Testing
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_image_features, test_labels, countries = extract_image_embeddings(test_dataloader, model, device)
    
    # Country labels
    train_labels = torch.load("./validations/country_labels.pt", weights_only=True)
    train_labels = train_labels.to(device)
    num_classes = len(torch.unique(train_labels))  # Total number of unique countries
    
    classifier = nn.Linear(512, num_classes).to(device)
    state_dict = torch.load("./training_new_classifier/classifier.pth", weights_only=True)  # Load the saved state dict
    classifier.load_state_dict(state_dict) 

    # Set the classifier to evaluation mode
    classifier.eval()

    top_1_correct = 0
    top_5_correct = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        outputs = classifier(test_image_features)

    for idx, output in enumerate(tqdm(outputs)):
        predicted = torch.argsort(output, descending=True)
        predicted_country = idx_to_country(country_2_idx, predicted[0].item())
        actual_country = countries[idx]
        
        top_5_countries = [idx_to_country(country_2_idx, pred.item()) for pred in predicted[:5]]

        if predicted_country == actual_country:
            top_1_correct += 1
        if actual_country in top_5_countries:
            top_5_correct += 1
            
    top1_accuracy = top_1_correct / len(test_dataloader)
    top5_accuracy = top_5_correct / len(test_dataloader)

    print("Total test samples: ", len(test_dataloader))
    print(f"Top-1 Accuracy: {top1_accuracy * 100:.1f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.1f}%")
    