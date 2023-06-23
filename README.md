# Neural_Network_Classifier_for_MNIST_Handwritten_Digits_Dataset
This program demonstrates the implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras for the task of recognizing handwritten digits from the MNIST dataset. The MNIST dataset consists of 60,000 training samples and 10,000 testing samples of grayscale images of handwritten digits (0-9). 

# Prerequisites
TensorFlow
Keras
NumPy
Matplotlib
Getting Started
# Import the required libraries:
TensorFlow: The main deep learning framework used in this program.
Keras: A high-level neural networks API that runs on top of TensorFlow.
Matplotlib: A plotting library for visualizing data and results.
NumPy: A library for numerical computations in Python.
# Load the MNIST dataset:

The MNIST dataset is imported from TensorFlow's tf.keras.datasets module and assigned to the variable mnist.
The data is divided into training and testing datasets, stored in the variables x_train, y_train, x_test, and y_test.
# Data Exploration:

Print the shape of the training data (x_train.shape) to see the dimensions of the training samples.
Print the label of the first image in the training data (y_train[0]).
Print the pixel values of the first image in the training data (x_train[0]).
# Data Preprocessing:

Set the dimensions and properties of the input images.
Reshape the training and testing data to match the expected input shape of the CNN.
Normalize the pixel values of the images between 0 and 1 using tf.keras.utils.normalize.
# Model Architecture:

Create a Sequential model using tf.keras.models.Sequential.
Add Convolutional layers to extract features from the images:
The first Conv2D layer has 32 filters and a 3x3 kernel size, using ReLU activation.
A MaxPooling2D layer follows with a 2x2 pool size.
Two additional Conv2D layers are added, each with increased filter count and ReLU activation.
Another MaxPooling2D layer reduces the spatial dimensions further.
Flatten the output from the Convolutional layers to prepare for the dense layers.
Add a Dense layer with 128 units and ReLU activation.
Apply dropout regularization to prevent overfitting.
Add the output layer with 10 units and softmax activation for multi-class classification.
# Model Compilation and Training:

Compile the model using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric.
Train the model using the training data and validate it using the testing data.
Visualize the training and validation accuracy over epochs using Matplotlib.
# Model Evaluation:

Evaluate the trained model on the testing data and print the test loss and accuracy.
# Image Prediction:

Select a new image from the testing data (x_test[new_image]) and display it using Matplotlib.
Use the trained model to predict the digit in the image (model.predict).
Print the predicted class label (pred.argmax()).

# Usage
a) Ensure that all the required libraries are installed and imported correctly.
b) Run the program in a Python environment that supports TensorFlow and Keras.
c) The program will load the MNIST dataset, preprocess the data, train the CNN model, and display the training and validation accuracy.
d) After training, the program will evaluate the model's performance on the testing data and print the test loss and accuracy.
d) Finally, it will display a selected image from the testing data and predict the corresponding digit.  
