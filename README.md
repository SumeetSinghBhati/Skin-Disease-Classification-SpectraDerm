# Skin Disease Classification using ResNet50

## Project Overview

This project involves developing a skin disease classification model using the ResNet50 architecture. The model is trained to classify images of skin diseases into eight categories. The dataset is organized into training and testing folders with an 80-20% split for training and validation. The goal is to build a robust model capable of accurately classifying different types of skin infections.

## Dataset Overview

The dataset contains images categorized into the following skin disease types:

- **Bacterial Infections**
  - Cellulitis
  - Impetigo

- **Fungal Infections**
  - Athlete's Foot
  - Nail Fungus
  - Ringworm

- **Parasitic Infections**
  - Cutaneous Larva Migrans

- **Viral Skin Infections**
  - Chickenpox
  - Shingles

The dataset is structured into `train_set` and `test_set` directories, with images resized to 224x224 pixels.

## Code Overview
Data Preparation:

Images are loaded from the dataset, resized to 224x224 pixels, and split into training and validation sets.
Labels are encoded and converted to one-hot vectors.

Model Creation:

A ResNet50 model is used with pre-trained weights from ImageNet, excluding the top layer.
Additional layers include a Global Average Pooling layer, a Dense layer with 512 units, and a Dense output layer with softmax activation for classification.
Model Training:

The model is trained for 12 epochs with a batch size of 32.
Training and validation losses and accuracies are plotted to visualize the performance.

Model Evaluation:

The trained model is evaluated on the test set.
A confusion matrix is generated to visualize the performance across different classes.

## Results

Training Accuracy: The model achieved 100% accuracy on the training set.
Validation Accuracy: The model achieved approximately 96.26% accuracy on the validation set.
Confusion Matrix: The confusion matrix shows the model's performance on the test set, with most classes predicted correctly.

## Insights

The model demonstrates effective learning, with training loss dropping significantly over epochs and validation accuracy stabilizing at around 96.26%.
The confusion matrix indicates that the model performs well across most classes, though there are some misclassifications.

## Deployment
The full-fledged model is deployed and accessible on Hugging Face Spaces. You can explore and interact with the deployed model at the following link:

[SpectraDerm on Hugging Face](https://huggingface.co/spaces/sumeetsinghbhati07/SpectraDerm)


