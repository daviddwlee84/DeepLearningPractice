# Transfer Learning in NLP

## Overview

Strategies for applying pre-trained language representations (mention in BERT paper)

* Feature-based
    * ELMo
* Fine-tuning
    * BERT

## Feature-based approach

### ELMo (Embeddings from Language Models)

Deep contextualized word representation model

* Complex characteristics of word use (e.g., syntax and semantics)
* How these uses vary across linguistic contexts (i.e., to model polysemy)

Feature

* Contextual: The representation for each word depends on the entire context in which it is used
* Deep: Combine all layers of a deep pre-trained neural network
* Character based: ELMo representations are purely character based, allowing the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training

## Fine-tuning approach

### BERT (Bidirectional Encode Representations from Transformers)

![BERT](https://jalammar.github.io/images/bert-transfer-learning.png)

## Links

* [AllenNLP ELMo](https://allennlp.org/elmo)

### Article

* [**The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)**](https://jalammar.github.io/illustrated-bert/)

### Paper

* [BERT](https://arxiv.org/abs/1810.04805)
