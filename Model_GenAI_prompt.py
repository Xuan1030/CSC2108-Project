import os, clip, torch, ssl, json

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
        
        self.prompts = torch.cat(self.prompts)
        print(f"Loaded {count} images")
            
            
    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]))
        prompt = torch.cat(clip.tokenize(self.prompts[idx]))
        region_or_country_label = self.labels[idx]
        return image, prompt, region_or_country_label



# class Imageregion_or_countryDataset(Dataset):
#     def __init__(self, dataset_folder, prompt_json, transform=None):
#         self.image_paths = []
#         self.prompts = []
#         self.transform = transform

#         with open(prompt_json, "r") as f:
#             generated_results = f.readlines()

#         for js in generated_results:
#             js = json.loads(js)
#             if not js["Error"]:
#                 self.image_paths.append(os.path.join(dataset_folder, js["region_or_country"], js["image"]))
#                 cur_prompt = f"Image of {js['region_or_country']}, with {js['front_features']} in the front, {js['middle_features']} in the middle, and {js['back_features']} in the background"
#                 self.prompts.append(cur_prompt)
                
#                 # Add augmented images with the same prompt
#                 augmented_image_path_prefix = os.path.join(dataset_folder, js["region_or_country"], js["image"].replace(".jpg", "_augmented"))
#                 # Iterate through augmented images
#                 for i in range(8):
#                     augmented_image_path = f"{augmented_image_path_prefix}_{i}.jpg"
#                     if os.path.exists(augmented_image_path):
#                         self.image_paths.append(augmented_image_path)
#                         self.prompts.append(cur_prompt)
            
                
#     def __len__(self):
#         return len(self.images)
    
    
#     def __getitem__(self, idx):
#         image = self.transform(Image.open(self.image_paths[idx]))
#         prompt = self.prompts[idx]
#         return image, prompt



def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 



def find_image_index(dataset, query_image_tensor):
    for idx, (image, _) in enumerate(dataset):  # Iterate through dataset
        if torch.equal(image, query_image_tensor):  # Compare tensors
            return idx
    return -1  # Return -1 if no match is found



def train_clip(train_dataloader, val_datakoader, model, epochs, learning_rate, device):    
    model.to(device)
    model.train()

    # set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.98), eps=1e-6, weight_decay=1e-4)
    img_loss = nn.CrossEntropyLoss()
    txt_loss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        
        process_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for images, prompts, labels in process_bar:
            
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
    torch.save(model.state_dict(), f"./training_clip_with_prompt/training_prompt_clip_model_epoch_{epoch+1}.pt")
    return model
        


# Ensure multiprocessing safety
if __name__ == "__main__":
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For macbook
    # device = "mps"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    # Apply CLIP on the datasetß
    dataset_folder = "datasets/compressed_dataset"
    dataset_pt_file = "training_clip_with_prompt/prompted_dataset.pt"
    
    prompt_json = "parsed_responses.jsonl"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    if os.path.exists(dataset_pt_file):
        ds = torch.load(dataset_pt_file)
    else:
        ds = ImagePromptDataset(dataset_folder, prompt_json, transform=transform)
        torch.save(ds, dataset_pt_file)
    
    train_size = int(0.8 * len(ds))
    train_dataset, val_dataset = random_split(ds, [train_size, len(ds) - train_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Training the model
    # model = train_clip(train_dataloader, val_dataloader, model, 3, 1e-6, device)
    
    # Traditional Validation
    # This part does not work very well atm...
    # val_process_bar = tqdm(val_dataloader, total=len(val_dataloader))
    # val_correct = 0
    # for images, prompts, labels in val_process_bar:
    #     images = images.to(device)
    #     prompts = prompts.to(device)
        
    #     logit_image, logit_text = model(images, prompts)
    #     predicted_labels = torch.argmax(logit_image, dim=1)
    #     for i, predicted_label in enumerate(predicted_labels):
    #         if predicted_label == labels[i]:
    #             val_correct += 1
        
    # val_accuracy = val_correct / len(val_dataset)
    # print(f"Validation accuracy: {val_accuracy}")


    # Validation using our scenario
    val_process_bar = tqdm(val_dataloader, total=len(val_dataloader))
    val_correct = 0
    for images, prompts, labels in val_process_bar:
        images = images.to(device)
        prompts = prompts.to(device)
        
        logit_image, logit_text = model(images, prompts)
        predicted_labels = torch.argmax(logit_image, dim=1)
        for i, predicted_label in enumerate(predicted_labels):
            if predicted_label == labels[i]:
                val_correct += 1
        
    val_accuracy = val_correct / len(val_dataset)
    print(f"Validation accuracy: {val_accuracy}")
    
    
    
    
    
    
