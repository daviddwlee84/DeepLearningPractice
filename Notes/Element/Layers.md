# Layers

## Overview

Table of content

* CNN
  * [Convolution Layer](#Convolution-Layer)
  * [Pooling Layer](#Pooling-Layer)
    * Pooling over (through) time
      * Max-over-time pooling
      * Average-over-time pooling
    * Local pooling
      * (Local) Max pooling
      * (Local) Average pooling
* [Fully-connected (Dense, Linear) Layer](#Fully-Connected-Layer-(Dense-Layer))
* RNN
  * LSTM
  * RNN
* Technique
  * Normalization
    * [BatchNorm](#batch-normalization-batchnorm)
  * Regularization
    * [Dropout Layer](#Dropout-Layer)
  * Gated Units used Vetrically
    * Residual Block
    * Highway Block

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

Covolution Kernel aka. Filter, Feature Map (i.e. the Weights)

> * usually have multiple "channels" - hope filters specialized in different things to gain different latent feature

Convolution without padding will shrink the output length based on the window size

### Dilation Convolution

> * another way of compressing data
> * to see a bigger spread of the sentence without having many parameters

## Pooling Layer

aka. Subsampling Layer

![pooling layer gif](https://mlnotebook.github.io/img/CNN/poolfig.gif)

* Reduce numbers of feature
* Prevent overfitting

Back Propagation

* [Only Numpy: Understanding Back Propagation for Max Pooling Layer in Multi Layer CNN with Example and Interactive Code. (With and Without Activation Layer)](https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4)

### Pooling Method

* **Max Pooling**: Pooling using a "max" filter with stride equal to the kernel size
* **Average Pooling**

> * In NLP, max pooling is better, because a lot of signals in NLP are sparse!

### Pooling Area

* **Pooling Over (Through) Time** (Over-time Pooling)
  * Max-over-time Pooling: capture most important activation (maximum over time)
* **Local Pooling**
  * Local Max Pooling
* K-Max Pooling Over Time
  * keep the orders of the "maxes"

> * In CV, pooling is normally mean local pooling
> * In NLP, pooling is normally mean pooling over time (global pooling)

## Fully Connected Layer (Dense Layer, Feed-forward Layer, Linear Layer)

## Technique

### Batch Normalization (BatchNorm)

> normalization

* Transform the output of a batch by scaling the activations to have **zero mean** and **unit variance** (i.e. **standard deviation of one**) => Z-transform of statistics

> * often used in CNNs
> * updated per batch so fluctuation don't affect things much
> * Use of BatchNorm makes models much *less sensitive to parameter initialization* (since outputs are automatically rescaled)
> * Use of BatchNorm also tends to make tuning of learning rates simpler

### Dropout Layer

> a regularization technique => deal with overfitting

* Create masking vector $r$ of Bernoulli random variables with probability $p$ (a hyperparameter) of being 1
* Delete features during training

$$
r \circ z
$$

* At test time, no dropout, scale final vector by probability $p$

$$
\hat{W}^{(S)} = p W ^{(S)}
$$

Reasoning: Prevents co-adaptation (overfitting to seeing specific feature constellations)

=> Usually Dropout gives 2~4% accuracy improvement!

> [CS224n 2019 Assignment 3 1-(b) Dropout](https://github.com/daviddwlee84/Stanford-CS224n-NLP/blob/master/Assignments/a3/written/assignment3.pdf) (Another scaling method (scale during training))

### Gated Units used Vetrically

> like the gating/skipping in LSTM and GRU

Key idea: summing candidate update with *shortcut connection* (is needed for very deep networks to work)

* Network: $F(x) = \operatorname{Conv}(\operatorname{ReLU}(\operatorname{Conv}(x)))$
* Identity $x$

#### Residual Block

$$
F(x) + x
$$

[Residual blocks — Building blocks of ResNet - Towards Data Science](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)

#### Highway Block

> $\circ$: Hadamard product (element-wise product)

$$
F(x) \circ T(x) + x \circ C(x)
$$

* T-gate: $T(x)$
* C-gate: $C(x)$

[Review: Highway Networks — Gating Function To Highway (Image Classification)](https://towardsdatascience.com/review-highway-networks-gating-function-to-highway-image-classification-5a33833797b5)

## Resources

* [ML Cheetsheet - Layers](https://ml-cheatsheet.readthedocs.io/en/latest/layers.html)
* [ML Cheetsheet - NN Concept - Layers](https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html#layers)

### Book

Dive Into Deep Learning

* Ch3.13 Dropout
* Ch5.4 Pooling Layer
* Ch5.10 BatchNorm

### Github

* [eriklindernoren/ML-From-Scratch - Layers](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py)
