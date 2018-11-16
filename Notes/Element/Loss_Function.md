# Loss Function

Table of Content

* Cross-Entropy
* Hinge
* Huber
* Kullback-Leibler
* MAE (L1)
* MSE (L2)

## Cross Entropy

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.

Cross-entropy loss increases as the predicted probability diverges from the actual label. A perfect model would have a log loss of 0.

**Formula**: (K class, y is one-hot vector, log is natural log)

$$
\operatorname{CE}(\mathbb{y}, \mathbb{\hat{y}}) = \displaystyle -\sum_{i=1}^K y_i \log(\hat{y}_i)
$$

## Resources

* [Loss Functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
* [Wiki - Loss function](https://en.wikipedia.org/wiki/Loss_function)
    * [Category:Loss functions](https://en.wikipedia.org/wiki/Category:Loss_functions)
    * [Loss functions for classification](https://en.wikipedia.org/wiki/Loss_functions_for_classification)
        * [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)