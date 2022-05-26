# Data Smoothing

## Background

### [N-gram](N-GramModel.md)

### [Data Sparsity](N-GramModel.md#Data-Sparseness)

## Overview of Smoothing Technique

Simple Smoothing

* Addictive smoothing
  * Add-one smoothing
* Held-out Estimation (留存估計)
  * Deleted Estimation / Two-way Cross Validation (刪除估計)
* Good Turing smoothing
* ... etc.

Combination Smoothing

* Interpolation smoothing (插值)
  * Jelinek-Mercer smoothing
* Katz smoothing (backoff) (退回模型)
* Kneser-Ney smoothing

## Simple Smoothing

> All the n-gram which didn't appear will have the same probability distribution.

### Add-one Smoothing

> Add one to frequency of each n-gram

#### Addictive Smoothing

> Add $\delta$ instead of one to frequency of each n-gram.
> (Typically, $0<\delta\leq1$)

### Held-out Estimation

> If the corpus is large, it's a good method.
> Since it need to split data into two set.

#### Deleted Estimation / Two-way Cross Valiation

> If the corpus is small

### Good Turing Smoothing

$$
p_{GT} (\text{an n-gram occcuring r times}) = \frac{(r+1)N_{r+1}}{N\cdot N_r}
$$

* [Wiki - Good–Turing frequency estimation](https://en.wikipedia.org/wiki/Good%E2%80%93Turing_frequency_estimation)

## Combination Smoothing

### Interpolation Smoothing

#### Jelinek-Mercer Smoothing

### Katz Smoothing (Backoff Model)

### Kneser-Ney Smoothing

## Links

* [**Slides - Standford NLP Lunch Tutorial: Smoothing**](https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf)
* [Wiki - Smoothing](https://en.wikipedia.org/wiki/Smoothing)
* [NLP 筆記 - 平滑方法(Smoothing)小結](http://www.shuang0420.com/2017/03/24/NLP%20%E7%AC%94%E8%AE%B0%20-%20%E5%B9%B3%E6%BB%91%E6%96%B9%E6%B3%95(Smoothing)%E5%B0%8F%E7%BB%93/)
