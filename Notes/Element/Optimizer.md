# Optimizer / Optimization Algorithm

> How learning rate decay

## Overview

* [Stochastic Gradient Descent](#stochastic-gradient-descent)
* [Adam Optimizer](#adam-optimizer)

## Learning Rate Decay

---

## Stochastic Gradient Descent

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}-\alpha \nabla_{\boldsymbol{\theta}} J_{\text {minibatch }}(\boldsymbol{\theta})
$$

## Momentum

### Exponentially Weighted Moving Average

* Also called exponentially weighted moving average.
* Exponential moving average, which using [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) technique, is in contrast to a simple moving average.

When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.

The decay is used to make the order samples decay in weight exponentially.

> Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999, etc.

```py
ema = tf.train.ExponentialMovingAverage(decay=0.998)
```

**Formula**:

$$
{\begin{aligned}v_{t}&=\beta v_{t-1}+(1-\beta )\theta_{t}\\[3pt]&=\beta v_{t-1}+\beta (1-\beta )\theta_{t-1}+(1-\beta )^{2}\theta_{t-2}\\[3pt]&=\beta \left[v_{t-1}+(1-\beta )\theta_{t-1}+(1-\beta )^{2}\theta_{t-2}+(1-\beta )^{3}\theta_{t-3}+\cdots +(1-\beta )^{t-1}\theta_{1}\right]+(1-\beta )^{t}\theta_{0}.\end{aligned}}
$$

* $v_t$: Forecasting value at time t (exponential smoothing result)
* $\beta$: decay
* $\theta_t$: Actual data value at time t (maybe with some bias)

Meaning of Decay

$v_t$ is exponentially average over $\displaystyle\frac{1}{1-\beta}$ round.

> e.g. decay ($\beta$) = 0.98 --> $\displaystyle\frac{1}{1-0.98} = 50$

$\displaystyle\frac{(1-\epsilon)^{\frac{1}{\epsilon}}}{\beta} = \frac{\frac{1}{e}}{\beta}$

($\epsilon = 1-\beta$)

> e.g. $\displaystyle0.98^{50} \simeq \frac{1}{e}$
> and $\displaystyle\frac{0.98^{50}}{0.98} \simeq 0.37$

That means it decay to $\frac{\frac{1}{e}}{\beta}$ by taking $\frac{1}{1-\beta}$ round.

> e.g. it takes about 50 rounds decay to 37% (we can generally say that we took EMA(50))

#### Bias Correction

TBD

## AdaGrad

> AdaGrad stands for Adaptive Gradient Algorithm

## RMSProp

> RMSProp stands for Root Mean Square Propagation

## Adam Optimizer

> * Adam stands for Adaptive Moment Estimation
> * Taking momentum and RMSProp and putting them together

Adam Optimization1 uses a more sophisticated update rule with two additional steps than Stochastic Gradient Descent.

1. First, **momentum** $m$: a rolling average of the *gradients*

    $$
    \begin{aligned} \mathbf{m} \leftarrow \beta_{1} \mathbf{m}+\left(1-\beta_{1}\right) \nabla_{\boldsymbol{\theta}} J_{\text {minibateh }}(\boldsymbol{\theta}) \\ \boldsymbol{\theta} \leftarrow \boldsymbol{\theta}-\alpha \mathbf{m} \end{aligned}
    $$

    where $\beta_1$ is a hyperparameter between 0 and 1 (often set to 0.9).

2. Second, **adaptive learing rates**: keeping track of $v$ - a rolling average of the *magnitudes of the gradients*

    $$
    \begin{array}{l}{\mathbf{m} \leftarrow \beta_{1} \mathbf{m}+\left(1-\beta_{1}\right) \nabla_{\boldsymbol{\theta}} J_{\text {minibatch }}(\boldsymbol{\theta})} \\ {\mathbf{v} \leftarrow \beta_{2} \mathbf{v}+\left(1-\beta_{2}\right)\left(\nabla_{\boldsymbol{\theta}} J_{\text {minibatch }}(\boldsymbol{\theta}) \odot \nabla_{\boldsymbol{\theta}} J_{\text {minibatch }}(\boldsymbol{\theta})\right)} \\ {\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}-\alpha \odot \mathbf{m} / \sqrt{\mathbf{v}}}\end{array}
    $$

    where $\odot$ and $/$ denote elementwise multiplication and division (so $z\odot z$ is elementwise squaring)
    and $\beta_2$ is a hyperparameter between 0 and 1 (often set to 0.99).

    Adam divides the update by $\sqrt{\mathbf{v}},$

## Resources

### Tutorial

* deeplearning.ai
  * Exponential Moving Average
    * [**Exponentially Weighted Average**](https://youtu.be/lAq96T8FkTw)
    * [**Understanding Exponentially Weighted Average**](https://youtu.be/NxTFlzBjS-4)
    * [**Bias Correction of Exponentially Weighted Average**](https://youtu.be/lWzo8CajF5s)
  * [Learning Rate Decay](https://youtu.be/QzulmoOg2JE)
  * [Gradient Descent With Momentum (C2W2L06) - YouTube](https://www.youtube.com/watch?v=k8fTYJPd3_I)
  * [RMSProp (C2W2L07) - YouTube](https://www.youtube.com/watch?v=_e-LFe_igno)
  * [Adam Optimization Algorithm (C2W2L08) - YouTube](https://www.youtube.com/watch?v=JXQT_vxqwIs)
* [Optimization Tricks: momentum, batch-norm, and more | Lecture 10 - YouTube](https://www.youtube.com/watch?v=kK8-jCCR4is)

### Article

* [TENSORFLOW GUIDE: EXPONENTIAL MOVING AVERAGE FOR IMPROVED CLASSIFICATION](http://ruishu.io/2017/11/22/ema/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html)

### Paper

* [[1412.6980] Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
* [[PDF] ADADELTA: An Adaptive Learning Rate Method - Semantic Scholar](https://www.semanticscholar.org/paper/ADADELTA%3A-An-Adaptive-Learning-Rate-Method-Zeiler/8729441d734782c3ed532a7d2d9611b438c0a09a)

### Book

* Dive Into Deep Learning
  * Ch7.2 Gradient Descent
  * Ch7.4 Momentum
  * Ch7.5 AdaGrad
  * Ch7.6 RMSProp
  * Ch7.7 AdaDelta
  * Ch7.8 Adam

### Github

* [eriklindernoren/ML-From-Scratch - Optimizers](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py)
