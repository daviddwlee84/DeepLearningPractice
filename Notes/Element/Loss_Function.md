# Loss Function

## Overview

### Table of Content

Classification

* Cross-Entropy
* NLL
* Hinge
* Huber
* Kullback-Leibler

Regression

* MAE (L1)
* MSE (L2)

Metric Learning

* Dice
* Contrastive
* N-pair
* Triplet

### Brief Description

Measuring "Distance" between the answer we expected and the true answer.

## Beneral

### Cross Entropy Loss

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.

Cross-entropy loss increases as the predicted probability diverges from the actual label. A perfect model would have a log loss of 0.

**Formula**: (K class, y is one-hot vector, log is natural log)

$$
\operatorname{CE}(\mathbb{y}, \mathbb{\hat{y}}) = \displaystyle -\sum_{i=1}^K y_i \log(\hat{y}_i)
$$

#### Binary Classification Problem

> The last output layer of neural network should be `sigmoid`

(Another definition of Cross-Entropy)

* [The cross-entropy error function in neural network - Question 2](https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks)

$$
\operatorname{CE}(\mathbb{y}, \mathbb{\hat{y}}) = \displaystyle -\sum_{i=1}^2 y_i \log(\hat{y}_i) = -y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)
$$

#### Multi-class Classification Problem

> The last output layer of neural network should be `softmax`

### Mean Square Error Loss

### Negative Log Likelihood Loss

NLL Loss vs. Cross Entropy Loss

```py
import torch
import torch.nn.functional as F

thetensor = torch.randn(3, 3)
target = torch.tensor([0, 2, 1])

# NLL Loss
sm = F.softmax(thetensor, dim=1)
log = torch.log(sm)
nllloss = F.nll_loss(log, target)

# CE Loss
celoss = F.cross_entropy(thetensor, target)

print(nllloss, 'should be equal to', celoss)
```

## Metric Learning

### Dice Loss

Dice Loss in Action

* [Dice Loss PR · Issue #1249 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/1249)

### Contrastive Loss

### Multi-class N-pair loss

### Triplet Loss

--

## Resources

* [Loss Functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
* [Wiki - Loss function](https://en.wikipedia.org/wiki/Loss_function)
  * [Category:Loss functions](https://en.wikipedia.org/wiki/Category:Loss_functions)
  * [Loss functions for classification](https://en.wikipedia.org/wiki/Loss_functions_for_classification)
    * [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)

### Article

* [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
* [機器/深度學習: 基礎介紹-損失函數(loss function) - Tommy Huang - Medium](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E4%BB%8B%E7%B4%B9-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-2dcac5ebb6cb)
* [損失函數的設計(Loss Function) - Cinnamon AI Taiwan - Medium](https://medium.com/@CinnamonAITaiwan/cnn%E6%A8%A1%E5%9E%8B-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-647e13956c50)

### Github

* [eriklindernoren/ML-From-Scratch - Loss Functions](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/loss_functions.py)

### Paper

* [[1905.10675] Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding](https://arxiv.org/abs/1905.10675)
* [[1909.05235] SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](https://arxiv.org/abs/1909.05235)
