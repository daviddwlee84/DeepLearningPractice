# Machine Translation

## Overview

> * Rule-based Approach
> * Corpus-based Approach

### History

* 1950s: Early Machine Translation
  * mostly *ruled-based* - using a bilingual dictionary to map words to their counterparts
* 1990s-2010: Statistical Machine Translation
  * learn a *probabilistic model* from data
  * $\argmax_y P(y|x) = \argmax_y\underbrace{P(x|y)}_{\text{Translation Model}}\underbrace{P(y)}_{\text{Language Model}}$
  * learning alignment: correspondence between particular words in the translated sentence pair
* 2014 after: Neural Machine Translation
  * [sequence-to-sequence](../Mechanism/seq-to-seq.md)

## Resources

* [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 8 – Translation, Seq2Seq, Attention - YouTube](https://www.youtube.com/watch?v=XXtpJxZBa2c&feature=youtu.be)
