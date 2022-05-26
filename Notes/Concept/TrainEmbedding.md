# Train Embedding

自然語言處理 10_ slides

## Prediction Learning Model

Prediction Learning Model Overview

* Training Data
  * Text = $w_1w_2 \dots w_T$
  * Sampling with a window
    * $(\text{begin}w_2, w_1) (w_1w_3, w_2), \cdots, (w_{T-2}w_T, w_{T-1}), (w_{T-1}\text{end}, w_T)$
  * Objective function - Average logarithm maximum likilihood function
    * Maximum Likilihood Estimation: Find a embedding that can maximum the objective function

Modeling with Neural Network

* Input layer
* Non-linear transform layer
* Linear transform layer
* Softmax output layer
* Pairwise rank lost



TBD: formula


### Collobert & Weston Model

* Language segment length (window) is 11 (i.e. l = 5)
* Non-linear activation function: HardTanh
* |V| = 30000
* Hidden layer dimension: 100
* Embedding dimension: 50
* Training corpus: Wikipedia with 631M words

Paper: [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)

### CBOW (Continuous Bag-of-Words)

> Simplifier: No hidden layer

Predict objective word with context.

* Input layer: Use "summation" to represent a bag-of-words
* Output layer


TBD

### SkipGram

> Even simplifier: No "summation"

* Input layer
* Sampling with window on unlabeled training data
  * $(w_1, w_2)$
* 

### Negative Sampling Training

> To simplify the output calucation (calculating softmax is too expensive)
>
> This training method can apply on not only SkipGram model but also any other model that use softmax as output layer (e.g. Collobert & Weston Model, CBOW Model)

**Negative Sampling** for a given word $w$.

* Get positive sample from training data sample $(w, c)$
* Random generate negative sample $(w, c')$






Construct negative sample

### Conclusion of CBOW vs. SkipGram

Predict

* CBOW: "predicting the word given its context"
* SkipGram: "predicting the context given a word"

Example "Hi fred how was the pizza?"

* CBOW with 3-grams: {"Hi fred how", "fred how was", "how was the", ...}
* Skip-gram with 1-skip and 3-grams: {"Hi fred how", "Hi fred was", "fred how was", "fred how the", ...}

## Matrix Factorization Learning Model

* Objective word $w$, Word in context $c$
  * $w \in V_w$
  * $c \in V_c$
  * $(w, c) \in D$ Trianing data
* Define Co-occurrence Matrix $M$
  * $|V_w|$ rows
  * $|V_c|$ columns
  * $M_{ij}$ represent relation between $w_i$ and $c_j$

Relation representation

* Naive: co-occurence count
  * $M_{ij} = \#(w_i, c_j)$
* [PMI Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
* PPMI Positive PMI

### SVD

* non-symmetric embedding based on SVD
  * $W_{SVD} = U_d \cdot \Sigma_d$
  * $C_{SVD} = V_d$
* symmetric embedding based on SVD
  * $W_{SVD} = U_d \cdot \sqrt{\Sigma_d}$
  * $C_{SVD} = V_d \cdot \sqrt{\Sigma_d}$

### GloVe

## Evaluation

### Synonymous Prediction - Word Analygy Task

### Similarity Evaluation - Word Similarity Task

## Resources

* [What are the continuous bag of words and skip-gram architectures?](https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures)
* [CS 224D: Deep Learning for NLP Lecture Notes](https://cs224d.stanford.edu/lecture_notes/notes1.pdf)
