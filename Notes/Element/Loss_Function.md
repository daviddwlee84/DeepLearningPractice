# Loss Function

## Overview

### Table of Content

* Cross-Entropy
* Hinge
* Huber
* Kullback-Leibler
* MAE (L1)
* MSE (L2)

### Brief Description

Measuring "Distance" between the answer we expected and the true answer.

## Cross Entropy

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.

Cross-entropy loss increases as the predicted probability diverges from the actual label. A perfect model would have a log loss of 0.

**Formula**: (K class, y is one-hot vector, log is natural log)

$$
\operatorname{CE}(\mathbb{y}, \mathbb{\hat{y}}) = \displaystyle -\sum_{i=1}^K y_i \log(\hat{y}_i)
$$

### Binary Classification Problem

(Another definition of Cross-Entropy)

* [The cross-entropy error function in neural network - Question 2](https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks)

$$
\operatorname{CE}(\mathbb{y}, \mathbb{\hat{y}}) = \displaystyle -\sum_{i=1}^2 y_i \log(\hat{y}_i) = -y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)
$$

## Resources

* [Loss Functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
* [Wiki - Loss function](https://en.wikipedia.org/wiki/Loss_function)
    * [Category:Loss functions](https://en.wikipedia.org/wiki/Category:Loss_functions)
    * [Loss functions for classification](https://en.wikipedia.org/wiki/Loss_functions_for_classification)
        * [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
* [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

### Github

* [eriklindernoren/ML-From-Scratch - Loss Functions](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/loss_functions.py)
