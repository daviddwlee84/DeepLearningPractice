# Attention

> Attention is a general deep learning *technique*

## Overview

### Concept

* Attention is **not** a part of Deep Learning
  * so we can calculate attention individually
* Attention is more like a "mechanism" of  the **weighted summation**

### History

* 2014 - Recurrent Models of **Visual** Attention
  * RNN model with attention for image classification
* 2014~2015 - Atention in Neural Machine Translation
* 2015~2016 - Attention-based RNN/CNN in NLP
* 2017 - Self-attention

![attention history](https://github.com/EvilPsyCHo/Attention-PyTorch/raw/master/pic/2_1.png)

### Improvements

* Attention significantly *improves Neural Machine Translate performance*
  * its useful to allow decoder to *focus on certain parts* of the source
* Attention *solves the bottleneck problem*
  * attention allows decoder to *look directly at source; bypass bottleneck*
* Attention *helps with vanishing gradient problem* (TODO link to the vanishing gradient, shortcut)
  * provides *shortcut* to faraway states
* Attention provides some *interpretability* (e.g. the visualization attention matrix)
  * By inspecting attention distribution, we can see what the decoder was focusing on

## Soft-alignment in Seq2seq model

* we have **encoder hidden states** $h_1, \dots, h_N \in \mathbb{R}^h$
* on timestep $t$, we have **decoder hidden states** $s_t \in \mathbb{R}^h$

And then, we get the attention score $e^t$ for this step

> [dot product](#basic-dot-product-attention) => get a scalar score

$$
e^t = [s_t^T h_1, \dots, s_t^T h_N] \in mathbb{R}^N
$$

We take *softmax* to get the *attention distribution* $\alpha^t$ for this step

> this is a probability distribution and sums to 1

$$
\alpha^t = \operatorname{softmax}(e^t) \in \mathbb{R}^N
$$

We use $\alpha^t$ to take a *weighted sum* of the encoder hidden states to get the **attention output** $a_t$

$$
a_t = \sum_{i=1}^N \alpha_i^t h_i \in \mathbb{R}^h
$$

Finally we *concatenate* the attention output $a_t$ with the decoder hidden state $s_t$

$$
[a_t;s_t] \in \mathbb{R}^{2h}
$$

## The "General Definition" of attention

Given a set of vector *values*, and a vector *query*, **attention** is a **technique** to compute a "weighted sum of the *values*", dependent on the *query*.

> query attends to the values

### Understand Attention as "Query"

* Values - a set of vectors
* Query - a single vector

**Intuition**:

* The weighted sum is a *selective summary* of the information contained in the *values*, where the *query* determines which values to focus on.
* Atteniton is a way to obtain a *fixed-size representation of an arbitrary set of representations* (the *values*), dependent on some other representation (the *query*)

## Several Attention Variants

We have

* Values $h_1, \dots, h_N \in \mathbb{R}^{d_1}$
* Query $s \in \mathbb{R}^{d_2}$

Attention **always involves**

1. Computing the **attention scores**: $e \in \mathbb{R}^N$ (there are multiple ways to do this)
   1. Basic dot-product attention
   2. Multiplicative attention
   3. Additive attention
2. Taking *softmax* to get **attention distribution** $\alpha$
3. Using attention distribution to take *weighted sum of values* thus obtaining the **attention output** $a$

### Basic dot-product attention

$$
e_i = s^T h_i \in \mathbb{R}
$$

* this assumes $d_1 = d_2$

### Multiplicative attention

$$
e_i = s^T W h_i \in \mathbb{R}
$$

* where $W \in \mathbb{R}^{d_2\times d_1}$ is a weight matrix

### Additive attention

$$
e_i = v^T \tanh(W_1h_i + W_2 s) \in \mathbb{R}
$$

* where $W_1 \in \mathbb{R}^{d_3\times d_1}$, $W_2 \in \mathbb{R}^{d_3\times d_2}$ are a weight matrices and $v \in \mathbb{R}^{d_3}$ is a weight vector
* $d_3$ (the attention dimensionality) is a hyperparameter

## Attention Area

### Global Attention vs. Local Attention

### Self-Attention

> intra-attention

* For each node/vector, create a query vector Q, key vector K and a value vector V

[Multi-head Self-attention Mechanism: Transformer](Transformer.md)

## Links

* [**Attention and Augmented Recurrent Neural Networks**](https://distill.pub/2016/augmented-rnns/)
  * [distillpub/post--augmented-rnns: Attention and Augmented Recurrent Neural Networks](https://github.com/distillpub/post--augmented-rnns)
* [successar/AttentionExplanation](https://github.com/successar/AttentionExplanation)

### Code

* [**Attention - PyTorch and Keras**](https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras): Introduce attention mechanism with example using PyTorch and Keras simultaneously

### Tutorial

* [Andrew Ng - C5W3L07 Attention Model Intuition](https://youtu.be/SysgYptB198)
* [Andrew Ng - C5W3L08 Attention Model](https://youtu.be/quoGRI-1l0A)
* [Youtube - Attention in Neural Network](https://youtu.be/W2rWgXJBZhU) - TODO
* [EvilPsyCHo/Attention-PyTorch](https://github.com/EvilPsyCHo/Attention-PyTorch) - Good attention tutorial
* [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention — PyTorch Tutorials 1.2.0 documentation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

### Paper

* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
* [[1601.06823] Survey on the attention based RNN model and its applications in computer vision](https://arxiv.org/abs/1601.06823)
