# GeoGuessr Bot

## **Introduction**
This GeoGuessr Bot is designed to predict the geographical location of a given image, such as those sourced from Google Maps or similar platforms. Leveraging deep learning techniques, including convolutional neural networks (CNNs) and multi-modal models like CLIP, the bot identifies the most likely country based on visual cues.

This repository provides the tools and instructions needed to train, evaluate, and deploy the GeoGuessr Bot.

---

## **Features**
- **Country Prediction**: Predicts the country from an image.
- **Image Augmentation**: Supports image resampling (rotation, cropping, etc.) for robust training.
- **Pre-trained Models**: Uses transfer learning with models like CLIP or ResNet for accuracy and efficiency.
- **Hierarchical Classification**: First predicts the continent, then narrows down to the country level.
- **Deployment Ready**: Can be deployed as a REST API for real-time predictions.
