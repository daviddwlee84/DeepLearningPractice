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

### Benefits

* Attention is trivial to parallelize (attention is permutation invariant)

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

## Several Attention Variants (A Family of Attention Mechanisms)

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

### Multiplicative attention (Bilinear, Product form)

> two vectors mediated by a matrix

$$
e_i = s^T W h_i \in \mathbb{R}
$$

* where $W \in \mathbb{R}^{d_2\times d_1}$ is a weight matrix

Space Complexity: $O((m+n) k)$, $W$ is $k \times d$

### Additive attention (MLP form)

> kind of a shallow neural network

$$
e_i = v^T \tanh(W_1h_i + W_2 s) \in \mathbb{R}
$$

* where $W_1 \in \mathbb{R}^{d_3\times d_1}$, $W_2 \in \mathbb{R}^{d_3\times d_2}$ are a weight matrices and $v \in \mathbb{R}^{d_3}$ is a weight vector
* $d_3$ (the attention dimensionality) is a hyperparameter

Space Complexity: $O(mnk)$, $W$ is $k \times d$

### Evolution Attention example of FusionNet

1. Origianl version of Bilinear form attention $S_{ij} = c_i^T W q_j$
2. Reduce the rank and complexity by dividing it into the product of two lower rank matrices $S_{ij} = c_i^T U^T V q_j$
3. Make the attention distribution to be symmetric $S_{ij} = c_i^T W^T D W q_j$ (sill make sence of linear algebra term)
4. Stick the left and right half through a ReLU $S_{ij} = \operatorname{ReLU}(C_i^TW^T)D \operatorname{ReLU}(Wq_j)$

* Smaller space
* Non-linearity

Space Complexity: $O((m+n) k)$, $W$ is $k \times d$

### Summary

| Attention Name           | Alignment score function                                                                                                       | Citation     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| Content-base             | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \text{cosine}[\boldsymbol{s}_t, \boldsymbol{h}_i]$                         |
| Additive(*)              | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])$ | Graves2014   |
| Location-Base            | $\alpha_{t,i} = \text{softmax}(\mathbf{W}_a \boldsymbol{s}_t)$                                                                 | Bahdanau2015 |
| General (multiplicative) | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\mathbf{W}_a\boldsymbol{h}_i$                         | Luong2015    |
| Dot-Product              | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\boldsymbol{h}_i$                                     | Luong2015    |
| Scaled Dot-Product(^)    | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \frac{\boldsymbol{s}_t^\top\boldsymbol{h}_i}{\sqrt{n}}$                    | Vaswani2017  |

> * (*) Referred to as “concat” in Luong, et al., 2015 and as “additive attention” in Vaswani, et al., 2017.
> * (^) It adds a scaling factor 1/n‾√, motivated by the concern when the input is large, the softmax function may have an extremely small gradient, hard for efficient learning.

broader categories of attention mechanisms

| Attention Name    | Alignment score function                                                                                                                                                                          | Citation          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| Self-Attention(&) | Relating different positions of the same input sequence. Theoretically the self-attention can adopt any score functions above, but just replace the target sequence with the same input sequence. | Cheng2016         |
| Global/Soft       | Attending to the entire input state space.                                                                                                                                                        | Xu2015            |
| Local/Hard        | Attending to the part of input state space; i.e. a patch of the input image.                                                                                                                      | Xu2015; Luong2015 |

> * (&) Also, referred to as “intra-attention” in Cheng et al., 2016 and some other papers.

## Soft vs. Hard Attention

## Attention Area

### Global Attention vs. Local Attention

### Self-Attention

> intra-attention

**Re-represent** the word representing based on its context (neighbors).

* For each node/vector, create a query vector Q, key vector K and a value vector V

$$
A(Q, K, V) = \operatorname{softmax}{(Q K^T) \over \sqrt{d_k}} V
$$

* dot product ($Q \cdot K$): compute similarity
* $\sqrt{d_k}$: a **scaling factor** to make sure that the dot products don't blow up

[Multi-head Self-attention: Transformer](Transformer.md)

## Resources

* [**Attention? Attention!**](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
* [**Attention and Augmented Recurrent Neural Networks**](https://distill.pub/2016/augmented-rnns/)
  * [distillpub/post--augmented-rnns: Attention and Augmented Recurrent Neural Networks](https://github.com/distillpub/post--augmented-rnns)
* [Attention機制詳解（二）—— Self-Attention與Transformer](https://zhuanlan.zhihu.com/p/47282410)
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
