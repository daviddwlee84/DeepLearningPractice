# Word Representation

## Concept

"Meaning" in linguistic way: (denotational semantics)

signifier (symbol) <==> signified (idea or thing)

## Obtain Word Meaning

* Usable resources
  * corpus
  * search engine
  * wikipedia
* Usable clue
  * relationship between words

Problems with resources like [WordNet](https://wordnet.princeton.edu/):

* Great as a resource but *missing nuance* (only correct in some contexts)
* Missing new meanings of words
* subjective
* requires human labor to create and adapt
* can't comupute accurate word similarity

### Vector Representation

* one-hot vector
  * each index represent a word

> * sparse vector
>   * able to explain by human
>     * relative words will have some visible relation

* dense vector
  * hard to explain by human

#### Representing words as discrete symbols

In traditional NLP, we regard words as discrete symbols - a localist representation

Words can be represented by **one-hot vector**

* vector dimension = number of words in vocabulary
* disadvantages:
  * vectors are *orthogonal*, that is no natural notion of **similarity** for one-hot vectors

Possible solution:

* try to rely on WordNet's list of synonyms
* instead, learn to encode similarity in the vectors themselves

#### Representing words by their context

> Distributional semantics: a word's meaning is given by the words that frequently appear close-by



## Word Similarity based on Context

* Context Vector

* [brightmart/nlu_sim](https://github.com/brightmart/nlu_sim) - all kinds of baseline models for sentence similarity

### Distributional Similarity

### Mutual Information

Pointwise Mutual Information (PMI)

* [Wiki - Pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)

### Similarity

* cosine
  * [Wiki - Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* Jaccard
  * [Wiki - Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
* Dice
  * [Wiki - Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
* JS
  * [Wiki - Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)

#### Word Mover's Distance

* [WMD_tutorial](https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html)
* [Word Mover’s Distance for Text Similarity - Towards Data Science](https://towardsdatascience.com/word-movers-distance-for-text-similarity-7492aeca71b0)

## Explicit Semantic Analysis

Wikipedia-based ESA

* [Wiki - Explicit semantic analysis](https://en.wikipedia.org/wiki/Explicit_semantic_analysis)

## Latent Semantic Analysis

> earlier than ESA

> human usually can't understand the meaning of the vector

Application => SVD

* [Wiki - Latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

## Page-count-based Similarity Scores (Cooccurance Measures)

## Word Sense Disambiguation 詞義消岐

## Word Sence Discrimination 詞義區分

### Application

* Machine Translation
* Information Retrieval
* Question Answering
* Knowledge Acquisition

### Lesk Algorithm

* [Wiki - Lesk algorithm](https://en.wikipedia.org/wiki/Lesk_algorithm)

#### Simulated Annealing

TBD

pun 雙關

homographic pun 語義雙關


[Embedding/Chinese-Word-Vectors: 100+ Chinese Word Vectors 上百种预训练中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)
