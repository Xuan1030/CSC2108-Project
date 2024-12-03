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
- **Baseline Training with unified Prompt**: Fine-Tune CLIP using unified text prompt "An image of {country_name}". Validation Accuracy = 81.0% top-5, 49% top-1
- **Fine-Tune with environment Prompt**： Fine-Tune CLIP using text prompt "An image of {country_name}, with {env_feature} and has {arc_feature}." Val acc=38.8902% top-1 70.26% top-5
- **Fine-Tune with feature Prompt**： Fine-Tune CLIP using text prompt "An image of {country_name}, with {feature1, feature2} in the frond, {feature3, feature4} in the middle and {feature5, feature6} in the back." Val acc = 41.7% top-1, 73.2% top-5
