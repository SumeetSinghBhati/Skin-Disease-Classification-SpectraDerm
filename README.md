Skin Disease Classification Using ResNet50
This project involves building a deep learning model to classify skin diseases using the ResNet50 architecture. The dataset includes various skin conditions, and the model is trained to accurately predict these conditions based on input images.

Table of Contents
Project Overview
Dataset
Installation
Usage
Model Training
Model Evaluation
Deployment
Insights
Project Overview
This project utilizes the ResNet50 model, a powerful convolutional neural network, to classify images of skin diseases into one of eight categories. The dataset is divided into training and testing sets, with a distribution ratio of 80% for training and 20% for testing.

Dataset
The dataset is organized into two main folders: train_set and test_set. The train_set folder contains images categorized into different types of skin diseases, including:

Bacterial Infections:

Cellulitis
Impetigo
Fungal Infections:

Athlete's Foot
Nail Fungus
Ringworm
Parasitic Infections:

Cutaneous Larva Migrans
Viral Skin Infections:

Chickenpox
Shingles
Each category is well-represented to ensure a balanced dataset for training the model.

Installation
To run this project, you need to install the following Python packages:

bash
Copy code
pip install opencv-python numpy tensorflow matplotlib seaborn scikit-learn
Usage
Prepare the Data: Load and preprocess the images from the dataset. Images are resized to 224x224 pixels and preprocessed for the ResNet50 model.

Model Training:

The ResNet50 model is used with ImageNet weights and fine-tuned with additional layers.
Training is conducted over 12 epochs with a batch size of 32.
Model Evaluation:

Evaluate the model on a validation set and plot training/validation loss and accuracy.
Test the model on unseen data and generate a confusion matrix to visualize performance.
Model Training
The model is trained using the following parameters:

Epochs: 12
Batch Size: 32
Optimizer: Adam
Loss Function: Categorical Crossentropy
The training results show high accuracy and low loss, indicating effective learning and minimal overfitting.

Model Evaluation
The model's performance is evaluated using accuracy and loss metrics on both training and validation sets. A confusion matrix is generated to analyze the classification performance on the test data.

Deployment
The full-fledged application is deployed on Hugging Face Spaces. You can interact with the deployed model and test it on new images using the following link:

SpectraDerm on Hugging Face

Insights
High Accuracy: The model achieves near-perfect accuracy on the training set and high accuracy on the validation set, demonstrating strong performance.
Confusion Matrix: The confusion matrix highlights the model's ability to correctly predict most classes, though some misclassifications are present.
Deployment: The model is deployed and accessible online, allowing users to test the model's performance interactively.
For any questions or feedback, feel free to reach out or contribute to the project.
