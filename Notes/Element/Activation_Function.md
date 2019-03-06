# Activation Function

Table of Content

* Linear
* Sigmoid
* Hyperbolic Tangent
* Rectified Linear Unit (ReLU)
* Leaky ReLU
* Softmax

## Concept

### Non-linearities

## Linear

## Sigmoid (Logistic Function)

It squashes a vector in the range (0, 1). It is applied independently to each element of $\vec{z}$

$$
\operatorname{sigmoid}(z_i) = \frac{1}{1 + \exp(z_i)}
$$

Gradient

$$
\frac{\partial E}{\partial w_i} = \sum_{d\in D}(t_d-o_d)\frac{\partial}{\partial w_i}(-o_d) = \cdots = \sum_{d\in D}(t_d-o_d)\operatorname{sig}(w\cdot x_d)(1-\operatorname{sig}(w \cdot x_d))\cdot x_{i,d}
$$

## Hyperbolic Tangent

* Tanh is just a rescaled and shifted sigmoid
* Tanh often performs well for deep nets

## Rectified Linear Unit

## Leaky ReLU

## Softmax

A "softened" version of the arg max.
A generalization of the sigmoid function. An exponential follow by normalization.

* **Soft**: continuous and differentiable
* **Max**: arg max (its result is represented as a [one-hot](https://en.wikipedia.org/wiki/One-hot) vector, is not continous or differentiable)

**Purpose**: To represent a probability distribution over a discrete variable with n possible values (over n different classes)

**Requirement**:

1. Each element of $\hat{y}_i$ be between 0 and 1
2. The entire vector sums to 1 (so that it represents a valid probability distribution)

**Approach**: (the same approach that worked for the Bernoulli distribution generalizes to the multinoulli distribution)

1. A linear layer predicts unnormalized log probabilities: (to be well-behaved for gradient-based optimization)
    $$
    \vec{z} = W^T \vec{h} + \vec{b}
    $$
    where $\vec{z}_i = \log \tilde{P}(y = i|\vec{x})$
2. Exponentiate and normalize $\vec{z}$ to obtain the desired $\hat{y}$
    $$
    \operatorname{softmax}(\vec{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
    $$

**Derivatives**:

$$
\frac {\partial \operatorname{softmax}({\vec {z}})_{i}}{\partial x_{j}}=\operatorname{softmax}({\vec {z}})_{i}(\delta _{ij}-\operatorname{softmax}({\vec {z}})_{j})
$$

>  use [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta) $\delta_{ij}$

## Resources

* [Wiki - Activation Function](https://en.wikipedia.org/wiki/Activation_function)
    * [Softmax function](https://en.wikipedia.org/wiki/Softmax_function)

### Article

* [Medium - Deep Learning: Overview of Neurons and Activation Functions](https://medium.com/@srnghn/deep-learning-overview-of-neurons-and-activation-functions-1d98286cf1e4)

Softmax

* [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

### Github

* [eriklindernoren/ML-From-Scratch - Activation Function](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py)

### Video

* [Youtube - Derivatives Of Activation Functions (C1W3L08)](https://youtu.be/P7_jFxTtJEo)

### Book

Deep Learning

* Ch 6.2.2.3 Softmax Units for Multinoulli Output Distributions
