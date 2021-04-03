# Language Model Concept

到時候要弄一下 這放的位置(README) 和內容都怪怪 重複
N元模型 算是其中的子類..

Table of Content

* Embedding
  * Word Embedding
    * Bag-of-Words (BOW)
    * Word2Vec
    * GloVe
    * FastText
  * Sentence Embedding
  * Context Embedding

* Model
  * BiLM (Bidirectional Language Model)
  * CNN for NLP
  * RNN for NLP

## Terminology

* Granularity 粒度: The granularity of data refers to the size in which data fields are sub-divided.
* Perplexity (ppl.) 困惑度
  * for evaluation

## Overview

### Definition

Language Modeling

* is the NLP task of *predicting what word comes next*
* given a sequence of words and compute the *probability distribution* of the next word (which can be any word in the vocabulary)

> * Input: sequence of words $x^{(1)}, x^{(2)}, \dots, x^{(t)}$
> * Output: prob dist of the next word $P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})$

Language Model

* a system that does language modeling
* assigns probability to a piece of text

### Life Example

> Language Modeling is a *subcomponent* of many NLP tasks

* input method (predictive typing)
* search engine
* speech recognition
* ...

### Building Language Model

* N-gram Language Model
* Neural Language Model
  * Fixed-window Neural Language Model
  * Recurrent Neural Networks (a family of neural architectures)

## Word Embedding

* Goal: Representing a word as a dense, low-dimensional, and real-valued vector.
* Input: Long documents (e.g. articles from Wikipedia) or Short texts (e.g. tweets).
* Output Word vectors.
* Usage: Initialization of deep architectures.
* Assumption: Words co-occur in a context are similar.

### Word2Vec

* Continous Bag-of-Words (CBOW): Predicting a target word (output) with the sum of the words in the context (input)
* Skip-gram: Predicting each of other words in the context (output) with the target word (input)

Both of these two model can be composed with three layers

> assume word set size is N, K hidden units

* Input Layer
  * input N-dimension one-hot layer
* Projection (Hidden) Layer
  * weight: $N \times K$
* Output Layer
  * softmax

* Speed-up: (because the softmax function has normalization term, it need to iterate through the entire word set)
  * **Hierarchical softmax**: Represent words as leaves of a binary tree.
    * Cost of objective calculaiton: V to log(V)
  * **Negative sampling**: Approximage the softmax with negative samples.
    * Cost of objective calculation: V to #

[Google word2vec](https://code.google.com/archive/p/word2vec/)

### GloVe (Global Vectors for Word Representation)

Learning word representations by factorizing a co-occurrence matrix

* [Stanford - GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
* [Github - stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe)

### FastText (Subword Model)

Modeling morphology of words by **embedding character n-grams** on top of the model of skip-gram

* [Github - facebookresearch/fastText](https://github.com/facebookresearch/fastText)

## CNN for NLP

* Goal: Representing a sequence of words (e.g. a sentence), or a similarity matrix (word-word similarity) as a dense, low-dimensional, and real-valued vector.
* Input: A matrix (e.g. each column is a word vector, or each element is a similarity score of a pair of words).
* Output: A vector (or a scalar).
* Advantage: Encodeing semantics of n-grams.

### CNN on a Natural Language Sentence

* Input: Word embedding
* Convolution: A filter slides a window on the word matrix with a stride size = 1
* Pooling (max): Reducing the result of convolution to a scalar

## RNN for NLP

* Goal: Representing a sequence of words (i.e. a sentnce) as dense, low-dimensional, and real-valued vectors.
* Input: A sequence of words (or characters).
* Output: A sequence of hidden states with each a representation of the sequence from the beginning to a secific position.
* Advantage: Encoding sequential relationship and dependency among words.

Architecture of RNN

* Similar to **Markov Chain**
* Learning an RNN with BPTT (Back Propagation Through Time) --> **Gradient Vanishing/Exploding** problem for long sequences
* The problem can be addressed by **defining f(．,．) with gates**
* Can stack may hidden layers by treating te hidden states in low layers as input

### RNN with LSTM

> RNN with separate **memory** (cell state)

* Input gate
* Forget gate
* Output gate
* Memory cell

### RNN with GRU

* Update gate
* Reset gate

> With a comparable performance with LSTM, but is much simpler (no cell state)

## Resources

* [Building Language Models](https://nlpforhackers.io/language-models/)

> this shouldn't put here

* [Vanishing / Exploding gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
  * Solutions:
    * Multi-level hierarchy
    * LSTM ([RNN - The Problem of Long-Term Dependencies](Notes/Technique/RNN.md#The-Problem-of-Long-Term-Dependencies))
    * Faster hardware
    * Residual networks
    * Other activation functions

### Tutorial

* [Andrew Ng - RNN W2L02 : Using Word Embeddings](https://www.youtube.com/watch?v=Qu-cvY4HP4g)
* [Andrew Ng - RNN W2L06 : Word2Vec](https://www.youtube.com/watch?v=jak0sKPoKu8)

CS224n

* [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 6 – Language Models and RNNs - YouTube](https://www.youtube.com/watch?v=iWea12EAu6U&feature=youtu.be)
