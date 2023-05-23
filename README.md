# CIFAR-10 Image Classification

This repository contains code for a CIFAR-10 image classification task. The CIFAR-10 dataset consists of 50,000 training images and 10,000 test images, categorized into 10 classes. The goal is to build and train a model that can accurately classify the images into their respective classes.

## Prerequisites

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn


## Code Overview

The [CNN_CIFAR_10.ipynb](CNN_CIFAR_10.ipynb) notebook demonstrates three different approaches to solve the CIFAR-10 image classification task: using a custom Multi-Layer Perceptron (MLP) block, using dense layers, and using a Convolutional Neural Network (CNN).

- The script starts by loading the CIFAR-10 dataset using the `tf.keras.datasets.cifar10.load_data()` function.
- It then pre-processes the data by reshaping and normalizing the input images.
- The MLP approach uses a custom `net` layer and a `MLP` layer to define a deep neural network. The `call` method of the `MLP` layer connects the layers sequentially and applies the ReLU activation function to each layer's output. The output layer uses the softmax activation function to produce class probabilities. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
- The dense layers approach uses the `Sequential` model from Keras to define a deep neural network with fully connected layers. The model is compiled and trained similar to the MLP approach.
- The CNN approach uses the `Sequential` model to define a convolutional neural network with convolutional and pooling layers. The model is compiled and trained similar to the other approaches.
- After training, the models are evaluated on the test dataset, and predictions are made on sample test images.
- The confusion matrix is generated and visualized using Seaborn's heatmap.

## Results

The performance of each model is evaluated using the accuracy metric.

- MLP approach: Achieved an accuracy of approximately 46.6% on the test set.
- Dense layers approach: Achieved an accuracy of approximately 47.2% on the test set.
- CNN approach: Achieved an accuracy of approximately 54.3% on the test set.

## Author

- Prajwal Dutta
- GitHub: [SciNoLimits](https://github.com/SciNoLimits)

Feel free to reach out if you have any questions or suggestions.
