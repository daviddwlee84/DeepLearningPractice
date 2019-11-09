# BERT: Bidirectional Encoder Representations from Transformers

Quick notes

* Learning contextual representaiton (similar to what ELMo did)
* Using the *encoder* from a Transformer network
* Can be fine-tuned with just one additional output layer
  * question answering
  * language inference

## Background

Problem: Language Models only use left context or right context, but language understanding is bidirectional

Why are LMs unidirectional?

1. Directionality is needed to generate a well-formed probability distribution
2. **Words can "see themselves"** in a bidirectional encoder

## Pre-training Tasks

> Train on **Wikipedia + BookCorpus**

1. Masked LM
2. Next Sentence Prediction

### 1. Masked LM

> Mask out $k\%$ of the input words, and then predict the masked words (where $k = 15\%$)

* Too little masking: Too expensive to train
* Too much masking: Not enough context

In detail:

Rather than always replacing the chosen words with [MASK] token

* 80% of the time: Replace the word with the [MASK] token
* 10% of the time: Replace the word with a random word
* 10% of the time: Keep the word unchanged
  * the purpose of this is to bias the representation towards the actual observed word.

### 2. Next Sentence Prediction

> To learn *relationship* between sentences

Predict whether:

1. `IsNextSentence`: Sentence B is actual sentence that proceeds Sentence A (A $\rightarrow$ B)
2. `NotNextSentence`: Sentence B is just a random sentence (A $\rightarrow$ random)

## Model

* Transformer encoder

2 model sizes

* BERT-Base: 12-layer (transformers), 768-hidden, 12-head
* BERT-Large: 24-layer, 1024-hidden, 16-head

### Sentence Pair Embedding

Input Embedding =

* **Token Embeddings** - Token embeddings are word pieces
* **Segment Embeddings** - Learned segmented embedding represents each sentence
  * Where the [CLS] to the first [SEP] will have the same embedding
  * And the second sentence till the ending [SEP] will have another same embedding
* **Positional Embedding** - Positional embedding is as for other Transformer architectures

## Fine-tuning

Simply learn a classifier built on the top layer for each task that you fine tune for

## Resources

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [**The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)**](https://jalammar.github.io/illustrated-bert/)
* [預訓練小模型也能拿下13項NLP任務，谷歌ALBERT三大改造登頂GLUE基準](https://mp.weixin.qq.com/s/kvSoDr0E_mvsc7lcLNKmgg)

### Tutorial

* [TDLS: BERT, Pretranied Deep Bidirectional Transformers for Language Understanding (algorithm)](https://youtu.be/BhlOGGzC0Q0) - TODO
* [BERT to the rescue! - Towards Data Science](https://towardsdatascience.com/bert-to-the-rescue-17671379687f)

### Application

* [terrifyzhao/bert-utils](https://github.com/terrifyzhao/bert-utils) - Generate sentence vector, document classification, document similarity

### Implementation

* [hanxiao/bert-as-service: Mapping a variable-length sentence to a fixed-length vector using BERT model](https://github.com/hanxiao/bert-as-service)
* [CyberZHG/keras-bert: Implementation of BERT that could load official pre-trained models for feature extraction and prediction](https://github.com/CyberZHG/keras-bert)
* [bojone/bert4keras: Our light reimplement of bert for keras](https://github.com/bojone/bert4keras)
* [kaushaltrivedi/fast-bert: Super easy library for BERT based NLP models](https://github.com/kaushaltrivedi/fast-bert)
* [brightmart/albert_zh: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS, 海量中文預訓練ALBERT模型](https://github.com/brightmart/albert_zh)
* [bhoov/exbert: A Visual Analysis Tool to Explore Learned Representations in Transformers Models](https://github.com/bhoov/exbert)
* [kaushaltrivedi/fast-bert: Super easy library for BERT based NLP models](https://github.com/kaushaltrivedi/fast-bert)
* [tomohideshibata/BERT-related-papers: BERT-related papers](https://github.com/tomohideshibata/BERT-related-papers)
