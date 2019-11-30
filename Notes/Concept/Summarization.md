# Summarization

## Overview

### Definition

* Given input text x, write a summary y which is shorter and contains the main information of x

### Background

#### Pre-neural summarization

* Mustly extractive
* Typically had a **pipeline**
  1. content selection: choose some sentences to include
     * sentence scoring functions
     * graph-based algorithms
  2. information ordering: choose an ordering of those sentences
  3. sentence realization: edit the sequence of sentences
     * simplify
     * remove parts
     * fix continuity issues

#### Neural summarization

First seq2seq summarization paper is in 2015.

## Category

### By Input

* Single-document
* Multi-document

### By Strategy

* Extractive summarization
* Abstractive summarization

#### Extractive summarization

> like highlighting

* *select parts* (typically sentences) of the original text to form a summary
* easier but restrictive (no paraphrasing)

#### Abstractive summarization

* *generate new text* using natural language generation techniques
* more difficult and more flexible (more human)

> single-document abstractive summarization is a translation task!

## Improvement from just a seq-to-seq attention model

### Copy Mechanism

> seq2seq systme are bad at copying over details (like rare words) correctly

We can use **attention** to copy words and phrases from the input to the output

* [[1609.07317] Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/abs/1609.07317)
* [[1602.06023] Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023)
* [[1603.06393] Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)
* [[1704.04368] Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
  * on each decoder step, calculate $p_{\text{gen}}$, the probability of generating the next word (rather than copying it)

### Global Content Selection

#### Bottom-up summarization

* Content selection stage: Use a sequence-tagging model to tag words as *include* or *don't-include*
* Bottom-up attention stage: The seq2seq+attention system can't attend to words tagged don't-include (apply a mask)

> Simple but effective

* Better overall content selection strategy
* Less copying of long sequences (i.e. more abstractive output)

> [[1808.10792] Bottom-Up Abstractive Summarization](https://arxiv.org/abs/1808.10792)

### Reinforcement Learning

Directly optimize ROUGE-L

* [[1705.04304] A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)
* [A Deep Reinforced Model for Abstractive Summarization Techniques - Salesforce.com](https://www.salesforce.com/products/einstein/ai-research/tl-dr-reinforced-model-abstractive-summarization/)

## Dataset

For Single-document summarization

* Gigaword
* LCSTS
* NYT, CNN/DailyMail
* Wikihow

For Sentence simplification

* Simple Wikipedia
* Newsela

## Evaluation

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

* ROUGE-1: unigram overlap
* ROUGE-2: bigram overlap
* ROUGE-L: longest common subsequence overlap

> Difference between BLEU
>
> * ROUGE is based on *recall* (but often use F1 version), while BLEU is based on *precision*
> * ROUGE scores are reported separately for each n-gram while BLEU is reported as a single number (which is combination of the precisions for 1~4-grams)

* [ROUGE: A Package for Automatic Evaluation of Summaries - ACL Anthology](https://www.aclweb.org/anthology/W04-1013/)
* [google-research/rouge at master · google-research/google-research](https://github.com/google-research/google-research/tree/master/rouge)

## Resources

* [mathsyouth/awesome-text-summarization: A curated list of resources dedicated to text summarization](https://github.com/mathsyouth/awesome-text-summarization)
* [A Survey on Neural Network-Based Summarization Methods](https://arxiv.org/pdf/1804.04589.pdf)
