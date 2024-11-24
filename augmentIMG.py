from PIL import Image, ImageEnhance
import random
import numpy as np
import os 

# Specify the directory
img_folder = "./compressed_dataset"

# Function to resample new images
def augment_image(image):
    augmented_images = []
    
    # Rotate
    for angle in [15, -15, 30, -30]:
        rotated_image = image.rotate(angle)
        augmented_images.append(rotated_image)
    
    # Flip
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append(flipped_image)
    
    # Scale
    width, height = image.size
    scaled_image = image.resize((int(width * 1.2), int(height * 1.2)))
    augmented_images.append(scaled_image)
    
    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.5)
    dark_image = enhancer.enhance(0.7)
    augmented_images.extend([bright_image, dark_image])
    
    return augmented_images

# Return True if we already augmented for at least once, False otherwise
def check_for_augmented_files(foldername, keyword="augmented"):
    augmented_files = [
        f for f in os.listdir(foldername) 
        if keyword in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if len(augmented_files) >= 1:
        return True
    else:
        return False

# Loop through each folder in the image folder
num_countries = 0
for foldername, subfolders, filenames in os.walk(img_folder):
    # Count the number of files in the current folder
    if foldername == "./compressed_dataset":
        continue
    
    num_files = len(filenames)
    # Augment country's images if it has less than 100 images
    if (num_files < 100): 
        if (check_for_augmented_files(foldername) == True):
            break
        else:
            for filename in os.listdir(foldername):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(foldername, filename)
                    image = Image.open(file_path)
                    resampled_images = augment_image(image)

                    # Save resampled images in the same folder as the original image
                    for i, img in enumerate(resampled_images):
                        # Generate a new filename for the augmented image
                        augmented_filename = os.path.join(foldername, f"{os.path.splitext(filename)[0]}_augmented_{i}.jpg")
                        img.save(augmented_filename)

for foldername, subfolders, filenames in os.walk(img_folder):
    # Count the number of files in the current folder
    if foldername == "./compressed_dataset":
        continue
    
    num_files = len(filenames)
    # Augment country's images if it has less than 100 images
    if (num_files < 100): 
        print(f"Folder Name: {foldername}, num_files:{num_files}")