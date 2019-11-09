# ELMo: Embeddings from Language Models

> the NAACL 2018 Best Paper

## Background

### Pre-ELMo - the TagLM

> Temp to improve the NER task (i.e. CoNLL 2003) and finally get 91.93% on F1 score

* use a stand alone pre-trained language model (weight frozen)
* concatenate the representation from LM with the supervised Bi-LSTM model

## The ELMo

* Why ELMo so popular:
  * a break through system which is not only improved on a single task (like TagLM only did on NER) but have siginificant improvement (3% of average) on different tasks.

> Salient features from AllenNLP website
>
> ELMo representations are:
>
> *Contextual*: The representation for each word depends on the entire context in which it is used.
> *Deep*: The word representations combine all layers of a deep pre-trained neural network.
> *Character based*: ELMo representations are purely character based, allowing the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training.

### Training the model

* Bidirectional LM
  * forward LM + backward LM individually
  * 2 layers of BiLSTM layers
* Use residual connection
* ...

$$
\operatorname{ELMo}_k^{task} = E(R_k;\theta^{task}) = \gamma^{task} \sum_{j=0}^L s_j^{task} h_{k, j}^{LM}
$$

* $\gamma^{task}$ (global scaling) scales overall usefulness of ELMo to task
* $s_j^{task}$ are softmax-normalized mixture model weights
  * a "weight for a level" which will multiply the "hidden state of a level" in each level

### Use with a task

1. First run BiLM to get representation for each word (i.e. the pre-training)
2. Then let (whatever) end-task model use them
   1. Freeze weights of ELMo for purposes of supervised model
   2. Concatenate ELMo weights into task-specific model (like TagLM did)

### Weighting of Layers

The two BiLSTM NLM layers have differentiated uses/meanings

* Lower layer is better for **low-level syntax**
  * e.g. POS tagging, syntactic dependencies, NER
* Higher layers is better for **higher-level semantics**
  * e.g. sentiment, semantic role labeling, question answering, SNLI

## Resources

* [**The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)**](https://jalammar.github.io/illustrated-bert/)
* [AllenNLP - ELMo: Deep contextualized word representations](https://allennlp.org/elmo)
  * [[1802.05365] Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
