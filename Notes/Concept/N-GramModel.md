# N-gram Model

## Concept

### Language Modeling

From the aspect of statistics, any sentence s in natural language is construct by any strings but with different probability.

* Given a natural language L, and a unknown probability distribution P(s).

Use a given language sample estimate P(s), this process is called *language modeling*.

$$
\sum_{s\in L} P(s) = 1
$$

#### Chain Rule

How to calculate P(s) for a given $s = w_1w_2\dots w_l$?

Apply *chain rule*,

$$
P(s)_l = P(w_1)P(w_2|w_1)P(w_3|w_1w_2)\dots P(w_l|w_1w_2\dots w_{l-1}) = \prod_{i=1}^lP(w_i|w_1w_2\dots w_{i-1})
$$

Example: John read a book

P(John read a book) = P(John) × P(read|John) × P(a|John read) × P(book|John read a)

* [Wiki - Chain rule](https://en.wikipedia.org/wiki/Chain_rule)
* [Wiki - Chain rule (probability)](https://en.wikipedia.org/wiki/Chain_rule_(probability))

### Markov Assumption

The probability of $w_i$ is only related to the previous $n-1$ words

$$
P(w_i|w_1w_2\dots w_{i-1}) = P(w_i|w_{i-n+1}w_{i-n+2}\dots w_{i-1})
$$

Only consider the slice constructed by n words. i.e. *n-gram*

$$
\operatorname{n-gram} = w_{i-n+1}w_{i-n+2}\dots w_{i-1}w_i
$$

Thus n-gram model is defined as

$$
P(s)_l = P(w_1)P(w_2|w_1)P(w_3|w_1w_2)\dots P(w_l|w_{l-n+1}w_{l-n+2}\dots w_{l-1}) = \prod_{i=1}^lP(w_i|w_{i-n+1}w_{i-n+2}\dots w_{i-1})
$$

* n=1, unigram
  * Example: John read a book
    * P(John read a book) = P(John) × P(read|John) × P(a|read) × P(book|a)
* n=2, bigram
* n=3, trigram

### How to choose N - History Information

For larger N

* More scenario information, more distinction
* More parameter, more calculation cost, more training corpus, parameter estimation is more unreliable

For smaller N

TBD

### Markov Model

* N-gram Model = n-1 Markov Process
* N-gram Model consider sentence as the outcome of Markov Process
  * Starting with the begin sign `<bos>`
  * Generate the words in the sentence
  * Until the end sign `<eos>`

* [Wiki - Markov chain](https://en.wikipedia.org/wiki/Markov_chain)

### How to construct N-gram Model

Data Preparation

* Find training corpus
* Tokenize corpus or segment
* Mark the edge of sentence by using special word `<bos>` and `<eos>`

Parameter Estimation

* Use training corpus to estimate parameter of model

### Calculate Probability of a Sentence

#### Relative Frequency

Let $c(w_1w_2\dots w_{n-1})$ as the times of n-gram $w_1w_2\dots w_{n-1}$ reveal in the corpus.

$$
P(w_n|w_1w_2\dots w_{n-1}) = \frac{c(w_1w_2\dots w_n)}{c(w_1w_2\dots w_{n-1})}
$$

#### Maximum Likelihood Estimation

TBD

$$
\theta_{ML} = \theta_{RF}
$$

### Data Sparseness

* n-gram is not in the training corpus => probability of the n-gram must be 0
* MLE given the unobserved event probability 0

#### Zipf's Law

* [Wiki - Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)

#### Add-one Smoothing

> alias Laplance Smoothing, Laplancian Smoothing, Additive Smoothing

smoothing -> discounting

everybody add-one -> a kind of discounting for the frequent words

solve the 0 prabability problem (data sparseness problem)

but it's not a good solution even though it's quite simple

there are some more different smoothing method

* [Additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)

### Entropy

In **information theory**, *entropy* describe the *average rate information* of a random variable

* [Wiki - Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))

#### Evaluation of Language Model - Cross Entropy

## Links

* [Wiki - N-gram](https://en.wikipedia.org/wiki/N-gram)
* [Stanford - N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
