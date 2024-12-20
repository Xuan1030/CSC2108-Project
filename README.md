# GeoGuessr Bot

## **Introduction**
This GeoGuessr Bot is designed to predict the geographical location of a given image, such as those sourced from Google Maps or similar platforms. Leveraging deep learning techniques, including convolutional neural networks (CNNs) and multi-modal models like CLIP, the bot identifies the most likely country based on visual cues.

This repository provides the tools and instructions needed to train, evaluate, and deploy the GeoGuessr Bot.

---

## **Features**
- **Country Prediction**: Predicts the country from an image.
- **Image Augmentation**: Supports image resampling (rotation, cropping, etc.) for robust training.
- **Pre-trained Models**: Uses transfer learning with models like CLIP or ResNet for accuracy and efficiency.
- **Deployment Ready**: Can be deployed as a REST API for real-time predictions.


## Training Notes
- **Base CLIP Model**: Performance of Pre-trained CLIP model. Top-5 Accuracy 59.3%, Top-1 Accuracy 34.1%
- **Neural Network on Top of CLIP**: Trained with one-layer network based on CLIP. Top-5 Accuracy: 88.0% Top-1 Accuracy: 61.4%
- **Pre-Train with clip and features**: Pre-train with Google Gemini generated image descriptions. Validation Accuracy = 68% top 5, 38% top-1
- **Baseline Training with unified Prompt**: Pre-train using unified text prompt "An image of {country_name}". Validation Accuracy = 81.0% top-5, 49.0% top-1

## Evaluations
### IM2GPS3K
- **Base CLIP Model**: 
- **Neural Network on Top of CLIP**: 
- **Pre-Train with clip and features**: 
- **Baseline Training with unified Prompt**: 47.9% top-5, 23.8% top-1

##