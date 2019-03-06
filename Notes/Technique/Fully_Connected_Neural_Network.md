# Fully Connected Neural Network

## Brief Description

> Alias or some similar terms
>
> * Multilayer Perceptron (MLP)
> * Dense Neural Network (DNN)
> * Deep Neural Network (DNN)
> * Artificial Neural Network (ANN)

Definition of ANNs

Properties of ANNs

* Many neuron-like *threshold switching units*
* Many *weighted* interconnections among units
* Highly parallel, distributed process
* Learning by adaption of the connection *weights*

* [Wiki - Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network)

### Neural Network Process

* [Forward Propagation](#Forward-Propagation) - Prediction phase
* [Back Propagation](#Back-Propagation) - Training phase

### Evolution

* [Perceptron (1st stage)](Perceptron.md)
  * too simple, that can't even represent a simple formula
* Multi-Layer Network (2nd stage) - this note
  * [XOR Perceptron](../../Project/PerceptronPractice)

## Multi-Layer

Mapping from space to space

### Hidden Layer

* Each hidden layer maps to a new feature space
* Each hidden node is a new constructed feature
* Original problem may become spearable (or easier)

## Forward Propagation

> Propagate activation from input to output layer

The input provide the initial information that then propagates up to the hidden units at each layer and finally produces output.

During training forward propagation can contiune onward until it produces a scalar cost $J(\theta)$

## Back Propagation

> Propagate errors from output to hidden layer

Back propagation (backprop) allows the information from the cost to then flow backwards through the network, in order to compute the gradient.

* Back-propagation is an algorithm that computes the *chain rule*, with a specific order of operations that is highly efficient.
* Back-propagation aka. chain rule is the procedure to compute gradients of the loss w.r.t. parameters in a multi-layer neural network. (to minimize a complicated function of the parameters)

### Training Algorithm

Three Protocols

* Batch Training
* Online Training
* Stochastic Training

#### Stochastic Backpropagation

#### Batch Backpropagation

## Others Notes

### On Training

* *No guarantee of convergence* (may *oscillate* or reach a local minima)
* In practice, many large networks can be trained on *large amounts of data* for realistic problems
* *Many epochs* (ten of thousands) may be needed for adequate training.
* *Termination criteria*
  * Number of epochs
  * Threshold on training set error
  * No decrease in error
  * Increased error on a validation set
  * ...
* To *avoid local minima*: several trails with different random initial weights with majority or voting techniques

### Choosing Learning Rate η

#### Adjusting η durning training

### Over-training Prevention, Overfitting Avoidance

TBD

Reason

* Using *too many hidden units* leads to over-fitting

Solution

* Keep an *hold-out validation* set and test accuracy after every epoch

#### Dropout Training

Each time decide whether to delete one hidden unit with some probability p.

## Resources

### Article

Overview

* [DIY AI: An old school matrix NN](https://towardsdatascience.com/diy-ai-an-old-school-matrix-nn-401a00021a55)

Back Propagation

* [**Back-Propagation is very simple. Who made it Complicated?**](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c) - An three layer FCNN with different activation function step by step!
    * [code](https://github.com/Prakashvanapalli/TensorFlow/blob/master/Blogposts/Backpropogation_with_Images.ipynb)
* [**Yes you should understand backprop**](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) - A much deeper description explain about why understand backprop is so important
* [A step by step back propagation example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

### Book

Deep Learning

* Back Propagation
    * Ch 6.5 Back-Propagation and Other Differentiation Algorithm
        * Ch 6.5.1 Computational Graphs
        * Ch 6.5.2 Chain Rule of Calculus
        * Ch 6.5.4 Back-Propagation Computation in Fully-Connected MLP
        * Ch 6.5.7 Example: Back-Propagation for MLP Training
