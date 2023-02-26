# Context-Free Syntax Analysis

## Syntax Analysis Overview

* "Word" as unit
  * Segmentation, Tokenization
  * Morphological analysis, Lemmatization, Stemming
  * Part-of-speech Tagging
  * ...
* "Sentence" as unit
  * Syntactic parsing
* "Paragraph" as unit
  * Discourse parsing

What is syntax?


## Background

### Chomsky Hierarchy

![Wiki Chomsky Hierarchy](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Chomsky-hierarchy.svg/640px-Chomsky-hierarchy.svg.png)

| Grammar | Languages                                           | Automaton                                       | Production rules (constraints)                                                                  |
| ------- | --------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Type-0  | **Recursively enumerable** (Natural Language)       | Turing machine                                  | $αAβ→β{\displaystyle \alpha A\beta \rightarrow \beta }$                                         |
| Type-1  | **Context-sensitive**                               | Linear-bounded non-deterministic Turing machine | $αAβ→αγβ\alpha A \beta \rightarrow \alpha \gamma \beta$                                         |
| Type-2  | [**Context-free**](#context-free-grammer) (**CFG**) | Non-deterministic pushdown automaton            | $A→α{\displaystyle A\rightarrow \alpha }$                                                       |
| Type-3  | **Regular**                                         | Finite state automaton                          | $A→a{\displaystyle A\rightarrow {\text{a}}}$ and $A→aB{\displaystyle A\rightarrow {\text{a}}B}$ |

> Production rules's symbols:
>
> * ${\displaystyle {\text{a}}}$ = terminal
> * $A$, $B$ = non-terminal
> * $\alpha$ , $\beta$ , $\gamma$  = string of terminals and/or non-terminals
>   * $\alpha$ , $\beta$  = maybe empty
>   * $\gamma$  = never empty

### Context-free Grammer

* [Wiki - Context-free grammar](https://en.wikipedia.org/wiki/Context-free_grammar)





...

parse 句法樹

LL

LR

e.g. Yacc




## Syntax Analysis Algorithm

* [Wiki - Parasing Algorithms](https://en.wikipedia.org/wiki/List_of_algorithms#Parsing)

Parsing Algorithm

* Top-down
  * LL
* Bottom-up
  * LR
    * Simple LR 標準LR
        > This can't be used on the language which has ambiguity and order => can't analysis natural language
    * Generalized LR 廣義LR
  * CYK
* Mixed/other
  * Earley

Big Picture

1. Generate all the Syntax Tree
2. Syntax Disambiguation, i.e. find the correct Syntax Tree

### Earley Algorithm

* [Wiki - Earley parser](https://en.wikipedia.org/wiki/Earley_parser)

Operations

* Prediction
* Scanning
* Completion

### LR ALgorithm

* [Wiki - LR parser](https://en.wikipedia.org/wiki/LR_parser)

#### Simple LR Algorithm

* [Wiki - Simple LR parser](https://en.wikipedia.org/wiki/Simple_LR_parser)

#### Generalized LR Algorithm

* [Wiki - GLR parser](https://en.wikipedia.org/wiki/GLR_parser)

--

## Statistics-based Syntax Analysis Algorithm（統計句法分析)

PCFG vs. CFG

For a PCFG, G is a quadruple

$$
G = (V_N, V_T, S, P)
$$

* $V_N$
* $V_T$
* $S$
* $P$

### CYK Algorithm

* [Wiki - CYK algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)

## Probabilistic CFG as Language Model

> Compare with **n-gram model**, PCFG-based language model considered the structure information of a sentence.
> N-gram model can be instead considered a sentence as a linear structure

### Probabilistic Context-Free Grammar in Detail

> aka. Stochastic CFG

* [Wiki - Probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar)

#### Basic questions of PCFG

### Inside and Outside Variable

* Inside Variable $\beta$
* Outside Variable $\alpha$

### Inside Algorithm

### Outside Algorithm

### Viterbi Algorithm

### Inside-Outside Algorithm

--

## Dependency Parsing (依存句法分析)

## Resources

* CS224n 2019 Lecture 5: Linguistic Structure: Dependency Parsing
