# Skin Disease Classification

This project is a skin disease classification model using a ResNet50-based convolutional neural network. The code includes data preparation, model training, and saving the trained model.

## Requirements

Before running the code, ensure you have the following libraries installed:

- OpenCV
- NumPy
- Matplotlib
- TensorFlow

You can install the required libraries using:

```bash
pip install opencv-python numpy matplotlib tensorflow
Code Overview
Data Preparation:

Loads images from the specified directory.
Resizes images to 224x224 pixels.
Splits data into training and validation sets.
Displays a sample of images with their labels.
Model Definition:

Uses ResNet50 as the base model with pre-trained weights.
Adds a GlobalAveragePooling2D layer followed by Dense layers.
Compiles the model with the Adam optimizer and categorical crossentropy loss function.
Data Processing:

Preprocesses images using preprocess_input from ResNet50.
Encodes labels and converts them to one-hot vectors.
Model Training:

Trains the model for 12 epochs with a batch size of 32.
Saves the trained model to a specified path.
