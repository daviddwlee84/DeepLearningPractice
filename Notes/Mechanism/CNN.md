# Convolution Neural Network

## Background

Inspire from *Receptive Field* of biology

Three main characteristic

* local connection
* shared weight
* sub-sampling during time and space (pooling)

## Overview

### Popular Framework

### Convolution Process Introduce

![convolution gif](https://mlnotebook.github.io/img/CNN/convSobel.gif)

## Terminology

* stride
* pedding
* step 步長
* window

Parameter sharing => extracting same characteristic feature

* filter (feature map) i.e. the weights

Layer

* Convolution Layer
* Pooling Layer
* Dropout Layer

## Concept

### Pedding

#### Narrow / Zero-padding

> padding = 0

Zero-padding is used so that the resulting image doesn't shrink.

#### Equal length

> padding = 1

#### Wide

> padding = 2

### Sub-Sampling (Pooling)

[Pooling Layer](../Element/Layers.md#Pooling-Layer)

Compress the result of the convolution

* Max
* Average
* Random
* Gaussian

Purpose

* *dimension reduction*
  * reduce parameter => reduce complexity of training
* reduce the chance of *overfitting*

## CNN Procedure

The Convolution Layer

* Sharing parameters/weight
  * Extracting same feature
  * Reduce the amount of parameters => the model can be deeper

The Pooling Layer

* Finding the *most significant* feature
* Even the output data length of convolution layer is not consistent, using pooling can transfer it into fixed length feature
* Losing the sequence structure message => but good at find some local feature that is *not relate with position*
  * e.g. It was "not good", .... The not good can be put in any position in sentence but still represent some sentiment.

Final FCNN

* Using the "feature extraction" by the previous convolution-pooling layers as input to train a classifier.

## CNN in NLP

```txt
Word Embedding --> Filters --> Pooling
```

In NLP, CNN is used to deal with sequential problem. Such as sentences. e.g. sentiment classification

### Pooling layer in NLP

* Max Pooling
* K-Max Pooling
* Chunk-Max Pooling

[自然語言處理中CNN模型幾種常見的Max Pooling操作](https://blog.csdn.net/malefactor/article/details/51078135)

#### Kim Model

* [paper - Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181)

## Popular Model

### LaNet-5

![LaNet-5](https://camo.githubusercontent.com/08a42fc04255ac31a7e3445ea4298a568cd0122f/68747470733a2f2f7777772e7265736561726368676174652e6e65742f70726f66696c652f48616f68616e5f57616e672f7075626c69636174696f6e2f3238323939373038302f6669677572652f66696731302f41533a33303539333931393936313038393440313434393935323939373930352f4669677572652d31302d4172636869746563747572652d6f662d4c654e65742d352d6f6e652d6f662d7468652d66697273742d696e697469616c2d617263686974656374757265732d6f662d434e4e2e706e67)

Structure:

* Input Layer: 1
* Hidden Layer: 5 (C1-S2-C3-S4-C5)
  * Convolution Layers
  * Pooling Layers
* Hidden Layer: 1
  * Fully Connected Layers
* Output Layer: 1

### AlexNet

* [Paper - ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [A Walk-through of AlexNet](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637)

### VGG16

### GoogLeNet (Inception Net)

* [Paper - Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
* [A Simple Guide to the Versions of the Inception Network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)

### ResNet (Residual Network)

* [Paper - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Resources

### Article

* [Convolutional Neural Networks - Basics](https://mlnotebook.github.io/post/CNN1/)
* [Convolutional Neural Networks - TensorFlow (Basics)](https://mlnotebook.github.io/post/tensorflow-basics/)

* [ResNet, AlexNet, VGGNet, Inception: Understanding various architectures of Convolutional Networks](https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/)

### Tensorflow

* [Build a Convolutional Neural Network using Estimators](https://www.tensorflow.org/tutorials/estimators/cnn)
  * [Github - cnn_mnist.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py)
