import os, clip, torch, ssl, json, shutil

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
class ImageLabelDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, area=None):
        self.images = []
        self.countries = []
        self.transform = transform
        self.area = area

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
        area_label = self.area

        image = Image.open(image_path).convert("RGB")
        if self.area is not None:
            text_prompt = f"A Street View photo from {area_label} area of {country_label}"
        else:
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



def train_clip(train_dataloader, model, epochs, learning_rate, device, save_path=None):    
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

        val_process_bar.set_description(f"Image {i+1}/{len(val_dataloader)}, Validation accuracy: {round(val_correct / (i+1) *100, 4)}%")
    val_accuracy = val_correct / len(val_dataloader)
    print(f"Validation accuracy: {val_accuracy}")
    return val_accuracy



def validate_2_stage_clip(val_dataloader, img_dataset_folder, urban_model, rural_model, device, k=5):
    # Validation using our scenario
    urban_model.eval()
    rural_model.eval()
    urban_model.to(device)
    rural_model.to(device)

    # Use original clip to test urban and rural
    original_clip, _ = clip.load("ViT-B/32", device=device)

    # Generate Country list
    country_list = [country for country in sorted(os.listdir(img_dataset_folder)) if '.' not in country]

    val_process_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    val_correct = 0

    for i, tup in val_process_bar:
        
        image, text_prompt, _, label = tup
        label = label[0]

        pred, _ = predict_image_urban_rural(image, original_clip, device, k=1)
        if "urban" in pred:
            model = urban_model
        elif "rural" in pred:
            model = rural_model

        predicted_labels, _ = predict_image(image, model, device, k)

        for predicted_label in predicted_labels:
            if predicted_label == label:
                val_correct += 1

        val_process_bar.set_description(f"Image {i+1}/{len(val_dataloader)}, Validation accuracy: {round(val_correct / (i+1) *100, 4)}%")
    val_accuracy = val_correct / len(val_dataloader)
    print(f"Validation accuracy: {val_accuracy}")
    return val_accuracy



def predict_image(image, model, device, k=5, img_dataset_folder="datasets/compressed_dataset"):
    model.eval()
    model.to(device)
    # Generate Country list
    country_list = [country for country in sorted(os.listdir(img_dataset_folder)) if '.' not in country]

    image = image.to(device)

    tokenized_prompts = torch.cat([clip.tokenize(f"A Street View photo from {country}", truncate=True) for country in country_list]).to(device)

    with torch.no_grad():
        image_encoded = model.encode_image(image)
        text_encode = model.encode_text(tokenized_prompts)

    # Calculate similarity
    image_encoded /= image_encoded.norm(dim=-1, keepdim=True)
    text_encode /= text_encode.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_encoded @ text_encode.T).softmax(dim=-1)

    values, indices = similarity[0].topk(k)
    predicted_labels = [country_list[idx] for idx in indices]

    return predicted_labels, values



def predict_image_urban_rural(image, model, device, k=5, img_dataset_folder="datasets/compressed_dataset"):
    model.eval()
    model.to(device)

    image = image.to(device)
    classes = ["urban", "rural"]
    
    tokenized_prompts = torch.cat([clip.tokenize(f"A street view photo of {label} area.", truncate=True) for label in classes]).to(device)

    with torch.no_grad():
        image_encoded = model.encode_image(image)
        text_encode = model.encode_text(tokenized_prompts)

    # Calculate similarity
    image_encoded /= image_encoded.norm(dim=-1, keepdim=True)
    text_encode /= text_encode.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_encoded @ text_encode.T).softmax(dim=-1)

    values, indices = similarity[0].topk(k)
    predicted_labels = [classes[idx] for idx in indices]

    return predicted_labels, values



def split_rural_urban(clip, dataset_folder, urban_path, rural_path, transform, device):
    '''
    Use CLIP to split the dataset into rural and urban images
    '''
    for foldername, subfolders, filenames in os.walk(dataset_folder):
        if foldername == dataset_folder:
            continue
        country = os.path.basename(foldername)
        
        urban_dir_path = os.path.join(urban_path, country)
        if not os.path.exists(urban_dir_path):
            os.makedirs(urban_dir_path)
        
        rural_dir_path = os.path.join(rural_path, country)
        if not os.path.exists(rural_dir_path):
            os.makedirs(rural_dir_path)
        
        for filename in tqdm(filenames):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(foldername, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)
            predicted_labels, values = predict_image_urban_rural(image, clip, device, k=1)
            if "rural" in predicted_labels:
                shutil.copy(img_path, rural_dir_path)
            elif "urban" in predicted_labels:
                shutil.copy(img_path, urban_dir_path)
            else:
                print(f"Could not classify {img_path} as rural or urban")



if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--split_images", action='store_true')
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--validate_2", action='store_true')
    parser.add_argument("--use_urban_dataset", type=str, default="training_2_stage_clip/urban_dataset.pt")
    parser.add_argument("--use_rural_dataset", type=str, default="training_2_stage_clip/rural_dataset.pt")
    parser.add_argument("--use_urban_model", type=str, default="training_2_stage_clip/urban_model.pt")
    parser.add_argument("--use_rural_model", type=str, default="training_2_stage_clip/rural_model.pt")
    parser.add_argument("--train_urban", action='store_true')
    parser.add_argument("--train_rural", action='store_true')
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--predict", type=str, default=None)

    args = parser.parse_args()
    # Check args
    if args.validate_2:
        if args.use_urban_model is None or args.use_rural_model is None:
            print("Urban or Rural model is not provided")
            exit()


    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For macbook
    # device = "mps"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    # Apply CLIP on the datasetß
    image_dataset_folder = "datasets/compressed_dataset"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # Call split function to create urban and rural datasets
    urban_path = "datasets/urban_images"
    rural_path = "datasets/rural_images"

    if args.split_images:
        # Use clip to split rural and urban datasets
        if not os.path.exists(urban_path):
            os.makedirs(urban_path)
        if not os.path.exists(rural_path):
            os.makedirs(rural_path)
        split_rural_urban(model, image_dataset_folder, urban_path, rural_path, transform, device)
    
    if args.predict is None:

        overall_val_sets = []
        if args.train_urban or args.validate or args.validate_2:
            # Load the urban and rural datasets
            if os.path.exists(args.use_urban_dataset):
                urban_dataset = torch.load(args.use_urban_dataset, weights_only=False)
            else:
                urban_dataset = ImageLabelDataset(urban_path, transform=transform, area="Urban")
                torch.save(urban_dataset, args.use_urban_dataset)
            
            urban_train_size = int(0.8 * len(urban_dataset))
            urban_train_dataset, urban_val_dataset = random_split(urban_dataset, [urban_train_size, len(urban_dataset) - urban_train_size])
            overall_val_sets.append(urban_val_dataset)

            urban_train_dataloader = DataLoader(urban_train_dataset, batch_size=32, shuffle=True)

            # Train urban model
            urban_model, _ = clip.load("ViT-B/32", device=device)
            if args.train_urban:
                print("Training urban model")
                urban_model = train_clip(urban_train_dataloader, urban_model, 3, 1e-6, device, save_path=args.use_urban_model)
            if args.validate:
                # Validate the urban model
                print("Validating urban model")
                urban_val_dataset.dataset.area = None
                urban_val_dataloader = DataLoader(urban_val_dataset, batch_size=1, shuffle=True)
                urban_model.load_state_dict(torch.load(args.use_urban_model, weights_only=True))
                urban_accuracy = validate_clip(urban_val_dataloader, urban_path, urban_model, device, args.k)    


        # Load the rural dataset
        if args.train_rural or args.validate or args.validate_2:
            # Same pipeline for rural
            if os.path.exists(args.use_rural_dataset):
                rural_dataset = torch.load(args.use_rural_dataset, weights_only=False)
            else:
                rural_dataset = ImageLabelDataset(rural_path, transform=transform, area="Rural")
                torch.save(rural_dataset, args.use_rural_dataset)

            rural_train_size = int(0.8 * len(rural_dataset))
            rural_train_dataset, rural_val_dataset = random_split(rural_dataset, [rural_train_size, len(rural_dataset) - rural_train_size])
            overall_val_sets.append(rural_val_dataset)

            rural_train_dataloader = DataLoader(rural_train_dataset, batch_size=32, shuffle=True)

            # Train rural model
            rural_model, _ = clip.load("ViT-B/32", device=device)
            if args.train_rural:
                print("Training rural model")
                rural_model = train_clip(rural_train_dataloader, rural_model, 3, 1e-6, device, save_path=args.use_rural_model)
            
            if args.validate:
                # Validate the rural model
                print("Validating rural model")
                rural_val_dataset.dataset.area = None
                rural_val_dataloader = DataLoader(rural_val_dataset, batch_size=1, shuffle=True)
                rural_model.load_state_dict(torch.load(args.use_rural_model, weights_only=True))
                rural_accuracy = validate_clip(rural_val_dataloader, rural_path, rural_model, device, args.k)

        if args.validate_2:
            overall_val_sets[0].dataset.area = None
            overall_val_sets[1].dataset.area = None
            val_dataset = torch.utils.data.ConcatDataset(overall_val_sets)
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
            
            urban_model, _ = clip.load("ViT-B/32", device=device)
            rural_model, _ = clip.load("ViT-B/32", device=device)
            
            urban_model.load_state_dict(torch.load(args.use_urban_model, weights_only=True))
            rural_model.load_state_dict(torch.load(args.use_rural_model, weights_only=True))

            validate_2_stage_clip(val_dataloader, image_dataset_folder, urban_model, rural_model, device, args.k)
    
    # Predict the image
    else:
        # Check if models exist
        if args.use_urban_model is None or args.use_rural_model is None:
            print("Urban or Rural model is not provided")
            exit()
        if not os.path.exists(args.use_urban_model) or not os.path.exists(args.use_rural_model):
            print("Urban or Rural model does not exist")
            exit()

        # Load the urban and rural models
        urban_model, _ = clip.load("ViT-B/32", device=device)
        urban_model.load_state_dict(torch.load(args.use_urban_model, weights_only=True))

        rural_model, _ = clip.load("ViT-B/32", device=device)
        rural_model.load_state_dict(torch.load(args.use_rural_model, weights_only=True))

        image = Image.open(args.predict).convert("RGB")
        image = transform(image).unsqueeze(0)
        predicted_labels, values = predict_image_urban_rural(image, model, device, k=1)
        
        if "urban" in predicted_labels:
            print("It is an Urban image, use urban model to predict")
            pred, values = predict_image(image, urban_model, device, k=5)

        elif "rural" in predicted_labels:
            print("It is a Rural image, use rural model to predict")
            pred, values = predict_image(image, rural_model, device, k=5)

        output_str = ""
        for label, value in zip(predicted_labels, values):
            output_str += f"{label}: {value.item()}\n"
        print("Predicted labels:")
        print(output_str)
        exit()
    
    
# Top-5 acc without prompt: 78.5 for urban, 80.5 for rural, overall 79.6
    

    
