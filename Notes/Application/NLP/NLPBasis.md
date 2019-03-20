# NLP Overview

Table of Content

Overview

NLP Task

* Classification/Disambiguation
  * Word-Sense Disambiguation (WSD) - polysemy
  * Coreference Resolution
* Sequence Labeling
  * Word Segmentation 分詞
    * [Word Segmentation]((../../Concept/ChineseWordSegmentation.md))
    * Sentences Segmentation
  * Information Extraction
    * Name Entity Recognition (NER)
    * Semantic Role Labeling
    * Part-of-Speech tagging 詞性/詞類標註
* Sequence Transformation
  * Machine Translations
* Structured Prediction
  * Parsing

## Overview

Many task in NLP can be reduced to disambiguation (classification).

Many sequence task can be reduced to sequence labeling. (then reduced to classification problem).

### Sequence Model

Probabilistic sequence models allow integrating uncertainty over multiple, interdependent classifcations and collectively determine the most linkely global assignment.

Probability Graphical model in this problem

* Hidden Markov Model (HMM): Directed Graphic Model
* Conditional Random Field (CRF): Undirected Graphic Model
* RNN

## Classification (Disambiguation)

> Disambiguation = Ambiguity Resolution

Choose the best solution from many potential candidates. Each candidate is reconized as a class.

### Word-Sense Disambiguation (WSD)

* [Wiki - Word-sense disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)

### Coreference Resolution

### Question Answering

## Sequence Labeling

* Each token in a sequence is assigned a label.
* Labels of tokens are dependent on the labels of other tokens in the sequence, particularly therir neighbors (not independent)

> input/output sequence length are the same

Sequence Segmentation

### Text Segmentation

#### Word Segmentation

* Each character is assigned a label 0/1.
* If the character is the end of a word, 1 is chosen, or 0 is assigned.

#### Sentences Segmentation

* If one word is the end of a sentence, 1 is chosen, or 0 is assigned.

### Information Extraction

#### Name Entity Recognition

Identifying names of *people*, *places*, *organizations*, etc. in text.

#### Semantic Role Labeling

#### Part-of-Speech Tagging (POS) Tagging

* [Part of speech](https://en.wikipedia.org/wiki/Part_of_speech)
* [Part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)

### Sequence Labeling as Classification

#### Sliding Window

Classify each token independently but use as input feature, information about the surrounding tokens.

#### Using Outputs as Inputs

Use category of either the preceding or succeeding tokens by going forward or back and using previous outout.

Forward Classification

Backward Classification

## Sequence Transformation

> input/output sequence length may be changed

### Machine Translation

## Structured Prediction

### Syntactic Parsing

* Dependency structure
* Constituency structure

#### Full Parsing

Parsing is an important task toward understanding natural languages

> Many NLP tasks do not require all information from parse trees

#### Shallow Parsing

Identifies only a subset of parse tree

> Shallow Parsing = Chunking (i.e. Named Entity Recognition (NER))

Modeling Shallow Parsing (as a sequence labeling problem)

* For each word label one of these:
  * B - Beggining
  * I - Inside a chunk but not a beginning
  * O - Outside a chunk

## Sequence Transduction

## Appendix

### Chinese part-of-speech table
