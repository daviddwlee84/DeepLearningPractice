# Layers

## Overview

Table of content

* BatchNorm
* [Convolution](#Convolution-Layer)
* Pooling
* Dropout
* Linear
* LSTM
* RNN

### General specking

![NN Concept - Layers](https://ml-cheatsheet.readthedocs.io/en/latest/_images/neural_network_simple.png)

#### Input Layer

Holds the data your model will train on. Each neuron in the input layer represents a unique attribute in your dataset (e.g. height, hair color, etc.).

#### Output Layer

The final layer in a network. It receives input from the previous hidden layer, optionally applies an activation function, and returns an output representing your model’s prediction.

#### Hidden Layer

Sits between the input and output layers and applies an activation function before passing on the results. There are often multiple hidden layers in a network. In traditional networks, hidden layers are typically fully-connected layers — each neuron receives input from all the previous layer’s neurons and sends its output to every neuron in the next layer. This contrasts with how convolutional layers work where the neurons send their output to only some of the neurons in the next layer.

#### Summary

Number of Layers of a NN = number of Hidden Layers + number of Output Layers

## Convolution Layer

## Resources

* [ML Cheetsheet - Layers](https://ml-cheatsheet.readthedocs.io/en/latest/layers.html)
* [ML Cheetsheet - NN Concept - Layers](https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html#layers)
