# Fully Connected Neural Network

## Brief Description

Alias or some similar terms

* Multilayer Perceptron (MLP)
* Dense Neural Network (DNN)
* Deep Neural Network (DNN)
* Artificial Neural Network (ANN)

### Quick View

Table TBD

## Neural Network Process

* Forward Propagation - Prediction phase
* Back Propagation - Training phase

### Forward Propagation

The input provide the initial information that then propagates up to the hidden units at each layer and finally produces output.

During training forward propagation can contiune onward until it produces a scalar cost $J(\theta)$

### Back Propagation

Back propagation (backprop) allows the information from the cost to then flow backwards through the network, in order to compute the gradient.

* Back-propagation is an algorithm that computes the *chain rule*, with a specific order of operations that is highly efficient.
* Back-propagation aka. chain rule is the procedure to compute gradients of the loss w.r.t. parameters in a multi-layer neural network. (to minimize a complicated function of the parameters)

## Resources

### Book

Deep Learning

* Back Propagation
    * Ch 6.5 Back-Propagation and Other Differentiation Algorithm
        * Ch 6.5.1 Computational Graphs
        * Ch 6.5.2 Chain Rule of Calculus
        * Ch 6.5.4 Back-Propagation Computation in Fully-Connected MLP
        * Ch 6.5.7 Example: Back-Propagation for MLP Training