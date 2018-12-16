# Chatbot

## Overview

* Non-Task-Oriented Dialogue System
    * Retrieval-base Method
    * Generation-base Method

## Retrieval-Based Chatbot

### Response Selction

#### Roadmap

* Message-Response Matching for Single-Turn Response Selection
    * Framework
        * Matching with sentance embeddings --extension--> Matching with external knowledge (Topic Aware Attentive RNN)
        * Matching with message-response interaction --extension--> Matching with multiple levels of representations (Knowledge Enhanced Hybrid NN)
            * Similarity matrix-based interaction
            * Attention-based interaction
    * Insights from empirical studies
    * Extension: Matching with external knowledge
* Context-Response Matching for Multi-Turn Response Selection
    * Framework
        * Embedding --> Matching
            * Dual-LSTM
            * Multi-view Response Selection Model
            * Deep Learning to Respond (DL2R)
        * Representation --> Matching --> Aggregation
            * Sequential Matching Network (SMN)
            * Sequential Attention Network (SAN)
    * Insights from empirical studies
* Other emerging research topics
    * Matching with better representations
    * Matching with unlabeled data

## Generation-Based Chatbots

* Sequence-to-sequence
    * Neural responding machine
    * Encoding-decoding framework
    * Model variants with different attention
* Attention mechanism
* Bi-directional modeling

## Datasets for Empirical Studies

* Ubuntu Dialogue Corpus - [1.0](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/), [2.0](https://github.com/rkadlec/ubuntu-ranking-dataset-creator), [paper](https://arxiv.org/abs/1506.08909), [kaggle](https://www.kaggle.com/rtatman/ubuntu-dialogue-corpus)

* Douban Conversation Corpus - [github](https://github.com/MarkWuNLP/MultiTurnResponseSelection)

## Evaluation Metrics

$R_n @k$: For each message, if the only positive response is ranked within top $k$ position of $n$ candidates, then $R_n @k = 1$. The final result is the average on messages.

## Resources

### Article

* [知乎 - 小哥哥，檢索式chatbot瞭解一下？](https://zhuanlan.zhihu.com/p/44539292)
* [檢索式人工智障識記](https://wyydsb.xin/other/chatbot.html)
