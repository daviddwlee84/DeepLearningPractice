# Optimizer / Optimization Algorithm

## Overview

## Exponential Moving Average

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

### Bias Correction

TBD

## Learning Rate Decay

---

## Adam

## Resources

### Tutorial

* deeplearning.ai
    * Exponential Moving Average
        * [**Exponentially Weighted Average**](https://youtu.be/lAq96T8FkTw)
        * [**Understanding Exponentially Weighted Average**](https://youtu.be/NxTFlzBjS-4)
        * [**Bias Correction of Exponentially Weighted Average**](https://youtu.be/lWzo8CajF5s)
    * [Learning Rate Decay](https://youtu.be/QzulmoOg2JE)

### Article

* [TENSORFLOW GUIDE: EXPONENTIAL MOVING AVERAGE FOR IMPROVED CLASSIFICATION](http://ruishu.io/2017/11/22/ema/)

### Tensorflow

* [`tf.train.ExponentialMovingAverage`](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)