# Sequence-to-Sequence (Encoder-Decoder) Model

> an example of Conditional Language Model
>
> * Language Model: the decoder is predicting the next word of the target sentence y
> * Conditional: its predictions are also conditioned on the source sentence x

* **Encoder**: Input sequence --> Hidden states of input (encoding of the input)
* **Decoder**: Hidden state of output --> Output sequence
  * Prediction of an output sequenc conditioned on an input sequence

Applications

* Machine translation (source language -> target language)
* Response generation in dialogue models
* Summarization (long text -> short text)
* Answer generation
* Paraphrase generation
* Dialogue (previous utterances -> next utterance)
* Parsing (input text -> output parse as sequence)
* Code generation (neural language -> code)

## General Concept

### Decoding

#### Greedy Decoding

generate (or "decode") the target sentence by *taking argmax on each step* of decoder

stopping criterion: keep decoding until the model produces a *\<END> token*

problem: there is no way to undo decisions

#### Exhaustive Search Decoding

try computing all possible sequences

> too expensive

#### Beam Search Decoding

core idea: on each step of decoder, keep track of the k most probable partial translation (which is called *hypotheses*)

(k is the *beam size*)

> in bean search decoding, different hypotheses may produce \<END> tokens on different timesteps

stopping criterion: when a hypothesis produces \<END>, that hypothesis is complete => place it aside and continue exploring other hypotheses via beam search

> usually we contine beam search until
>
> * reach timestep T
> * have at least n completed hypotheses
>
> (where T and n is pre-defined cutoff)

## Different Structures

### Neural Recurrent Sequence Models

### Recurrent Sequence to Sequence

### Convolutional-Based Sequence Models

### Transformer-based seq-to-seq

### Attention (The Encoder-Attention-Decoder) Architecture

**Attention**: "Pay attention to" different sub-sequences of the input when generating each of the token of the output

Modeling alignment in machine translation

### Bidirectional Encode Representations from Transformers (BERT)

Self-attention, borrowed from NMT, is powerful in representing contexts and responses

## Recources

### Tutorial

* [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 8 – Translation, Seq2Seq, Attention - YouTube](https://www.youtube.com/watch?v=XXtpJxZBa2c&feature=youtu.be)
* [Andrew Ng - RNN W1L02: Notation (of sequence model)](https://youtu.be/XeQN82D4bCQ)
