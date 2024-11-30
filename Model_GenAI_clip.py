import os, clip, torch, ssl, json

from argparse import ArgumentParser

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(0)

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
class ImagePromptDataset(Dataset):
    def __init__(self, dataset_folder, prompt_json, transform=None):
        self.image_paths = []
        self.prompts = []
        self.labels = []
        self.transform = transform

        count = 0
        with open(prompt_json, "r") as f:
            generated_results = f.readlines()

        for js in generated_results:
            js = json.loads(js)
            if not js["Error"]:
                cur_prompt = f"Image of {js['region_or_country']}, with {js['front_features']} in the front, {js['middle_features']} in the middle, and {js['back_features']} in the background"
                try:
                    assert len(cur_prompt.split()) <= 77
                except:
                    print(f"Error tokenizing prompt: {js}")
                    continue
                self.image_paths.append(os.path.join(dataset_folder, js["region_or_country"], js["image"]))
                self.prompts.append(cur_prompt)
                self.labels.append(js["region_or_country"])
                count += 1
                
                # Add augmented images with the same prompt
                augmented_image_path_prefix = os.path.join(dataset_folder, js["region_or_country"], js["image"].replace(".jpg", "_augmented"))
                # Iterate through augmented images
                for i in range(8):
                    augmented_image_path = f"{augmented_image_path_prefix}_{i}.jpg"
                    if os.path.exists(augmented_image_path):
                        self.image_paths.append(augmented_image_path)
                        self.prompts.append(cur_prompt)
                        self.labels.append(js["region_or_country"])
                        count += 1
        
        print(f"Loaded {count} images")
            
            
    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]))
        text_prompt = self.prompts[idx]
        prompt = clip.tokenize(text_prompt).squeeze(0)
        region_or_country_label = self.labels[idx]
        return image, text_prompt, prompt, region_or_country_label



class ImageLabelDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.images = []
        self.countries = []
        self.transform = transform

        for foldername, subfolders, filenames in os.walk(dataset_folder):
            if foldername == dataset_folder:
                continue
            country = os.path.basename(foldername)  # Get the folder name (country)
            
            for filename in filenames:
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(foldername, filename)
                self.images.append(img_path)
                self.countries.append(country) 
                
    def __len__(self):
        return len(self.countries)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        country_label = self.countries[idx]

        image = Image.open(image_path).convert("RGB")
        text_prompt = f"A Street View photo from {country_label}"
        prompt = clip.tokenize(text_prompt).squeeze(0)

        if self.transform:
            image = self.transform(image)
        return image, text_prompt, prompt, country_label



def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 



def find_image_index(dataset, query_image_tensor):
    for idx, (image, _) in enumerate(dataset):  # Iterate through dataset
        if torch.equal(image, query_image_tensor):  # Compare tensors
            return idx
    return -1  # Return -1 if no match is found



def train_clip(train_dataloader, model, epochs, learning_rate, device, save_path=None, use_prompt=True):    
    model.to(device)
    model.train()

    # set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.98), eps=1e-6, weight_decay=1e-4)
    img_loss = nn.CrossEntropyLoss()
    txt_loss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        
        process_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for images, _, prompts, _ in process_bar:
            
            images = images.to(device)
            prompts = prompts.to(device)
            
            optimizer.zero_grad()
            # Forward pass
            logit_image, logit_text = model(images, prompts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            # Compmute loss
            loss = (img_loss(logit_image, ground_truth) + txt_loss(logit_text, ground_truth)) / 2
            loss = loss.to(torch.float32)
            loss.backward()
            
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            process_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
        # Save model
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
        


def validate_clip_with_prompt(val_dataloader, img_dataset_folder, model, device, k=5):
    # Validation using our scenario
    model.eval()
    model.to(device)
    # Generate Country list
    country_list = [country for country in sorted(os.listdir(img_dataset_folder)) if '.' not in country]

    val_process_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    val_correct = 0

    for i, tup in val_process_bar:
        
        image, text_prompt, _, label = tup

        # Get text prompt since it contains feature keywords for image already
        text_prompt_without_label = text_prompt[0].split(",")[1]
        # Create fake prompts with all country names
        tokenized_prompts = torch.cat([clip.tokenize(f"Image of {country}, {text_prompt_without_label}", truncate=True) for country in country_list]).to(device)
        
        image = image.to(device)
        label = label[0]
        
        with torch.no_grad():
            image_encoded = model.encode_image(image)
            text_encode = model.encode_text(tokenized_prompts)

        # Calculate similarity
        image_encoded /= image_encoded.norm(dim=-1, keepdim=True)
        text_encode /= text_encode.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_encoded @ text_encode.T).softmax(dim=-1)

        values, indices = similarity[0].topk(k)
        predicted_labels = [country_list[idx] for idx in indices]

        for predicted_label in predicted_labels:
            if predicted_label == label:
                val_correct += 1

        val_process_bar.set_description(f"Image {i+1}/{len(val_dataset)}, Validation accuracy: {round(val_correct / (i+1) *100, 4)}%")
    val_accuracy = val_correct / len(val_dataset)
    print(f"Validation accuracy: {val_accuracy}")
    return val_accuracy



def validate_clip(val_dataloader, img_dataset_folder, model, device, k=5):
    # Validation using our scenario
    model.eval()
    model.to(device)
    # Generate Country list
    country_list = [country for country in sorted(os.listdir(img_dataset_folder)) if '.' not in country]

    val_process_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    val_correct = 0

    for i, tup in val_process_bar:
        
        image, text_prompt, _, label = tup

        # Create fake prompts with all country names
        tokenized_prompts = torch.cat([clip.tokenize(f"A Street View photo from {country}", truncate=True) for country in country_list]).to(device)
        
        image = image.to(device)
        label = label[0]
        
        with torch.no_grad():
            image_encoded = model.encode_image(image)
            text_encode = model.encode_text(tokenized_prompts)

        # Calculate similarity
        image_encoded /= image_encoded.norm(dim=-1, keepdim=True)
        text_encode /= text_encode.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_encoded @ text_encode.T).softmax(dim=-1)

        values, indices = similarity[0].topk(k)
        predicted_labels = [country_list[idx] for idx in indices]

        for predicted_label in predicted_labels:
            if predicted_label == label:
                val_correct += 1

        val_process_bar.set_description(f"Image {i+1}/{len(val_dataset)}, Validation accuracy: {round(val_correct / (i+1) *100, 4)}%")
    val_accuracy = val_correct / len(val_dataset)
    print(f"Validation accuracy: {val_accuracy}")
    return val_accuracy



if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--use_prompt", action='store_true')
    parser.add_argument("--use_dataset", type=str, default="training_clip_with_prompt/prompted_dataset.pt")
    parser.add_argument("--use_model", type=str, default="training_clip_with_prompt/training_prompt_clip_model_epoch_3.pt")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For macbook
    # device = "mps"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    # Apply CLIP on the datasetß
    image_dataset_folder = "datasets/compressed_dataset"
    dataset_pt_file = args.use_dataset
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Construct Dataset and DataLoader
    if args.use_prompt:
        prompt_json = "parsed_responses.jsonl"

        if os.path.exists(dataset_pt_file):
            ds = torch.load(dataset_pt_file, weights_only=False)
        else:
            ds = ImagePromptDataset(image_dataset_folder, prompt_json, transform=transform)
            torch.save(ds, dataset_pt_file)
    else:
        if os.path.exists(dataset_pt_file):
            print("loading local dataset")
            ds = torch.load(dataset_pt_file, weights_only=False)
        else:
            ds = ImageLabelDataset(image_dataset_folder, transform=transform)
            torch.save(ds, dataset_pt_file)
    

    train_size = int(0.8 * len(ds))
    train_dataset, val_dataset = random_split(ds, [train_size, len(ds) - train_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    

    # Training the model
    # model = train_clip(train_dataloader, model, 3, 1e-6, device, save_path="./training_clip_unified_prompt/model_epoch_3.pt")
    
    # Load the model
    model.load_state_dict(torch.load(args.use_model, weights_only=True))

    # Validate the model
    accuracy = validate_clip(val_dataloader, image_dataset_folder, model, device, args.k)
    
    
    
    
    
    
