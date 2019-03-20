# Chinese Word Segmentation

## Overview

Segmentation in English => Tokenization

* Abbreviation
* I'm, don't, Tom's, ...
* number, date, ...
* Word with "-"

Why segmentation?

* [Speech synthesis](https://en.wikipedia.org/wiki/Speech_synthesis)
  * e.g. 的（ㄉㄜ˙）, 目的(ㄉㄧˋ)
* Information retrieval
  * e.g. Search 機會 won't appear 飛"機會"飛
* Word analysis
  * Word frequency
  * ...

### Basic Method

* Word Table Based Method 基於詞表
  * Data driven, Rule driven
* Vocabulary Sequence Tagging Method 字序列標記

## Word Table Based Method

### Maximum Match 最大匹配

* MM/FMM: Match word table from left to right
* RMM: Match word table from right to left

Pros and cons

* pros
  * easy and simple
  * long word first
* cons
  * long word first not always right...

## Vocabulary Sequence Tagging Method

Vocabulary position tagging

* B: beginning
* M: middle
* E: end
* S: single word

Key problem

* Segmentation Disambiguation 切分歧義消解
* OOV (Out-of-vocabulary) 未登錄詞
  * not appear in word table
  * not appear in corpus

> [Word-Sense Disambiguation (WSD)](../Application/NLP/NLPBasis.md#Word-Sense-Disambiguation-(WSD))

### Disambiguation / Ambiguity Resolution

> ambiguous: open to more than one interpretation; having a double meaning

Types

* Intersaction disambiguation 交集型歧義
  * e.g. AJB (AJ, JB, A, B are all word) => AJ/B vs. A/JB
  * e.g. 從小學 => 從小/學, 從/小學
  * chain length (交集型歧義中交集字段的個數)
    * 從小學 => length = 1
    * 結合成分 => length = 2
    * 為人民工作 => length = 3
    * ...
* Combination disambiguation 組合型歧義
  * e.g. AB (AB, A, B are all word) => AB vs. A/B
* Mixed...

True ambiguity vs. Fake ambiguity

* True ambiguity
  * e.g. 地面積, 和平等, 把手
* Fake ambiguity: only become ambiguous when you check it independently from sentence
  * e.g. 挨批評, 平淡

> We can disambiguate only when we found ambiguity...

#### Bidirection MM (FMM + RMM)

* if the result of MM and RMM are not the same
* but it can't find all the cases
  * long word first
    * e.g. 馬上 will never be 馬/上
  * even chain length
    * e.g. 原子**結合成分子時**

* Find combinational ambiguity: MM + Reverse Minimal Match
* Find all ambiguity: All-segment algorithm (seperate all the possible combination)

Data Structure

Rule-based Disambiguation

Statistics-based Disambiguation

### OOV Recognization

#### Chinese Name Recognization

## Evaluation

* Precision (denominator: # of all the segmentation result)
* Recall (denominator: # of segmentation in answer)
* F1-score
* Use word as basic unit

### Banchmark

[SIGHAN](http://sighan.cs.uchicago.edu/)

### Standard

* China：信息处理用现代汉语分词规范 (GB/T13715-92) 1993
* Taiwan：資訊處理用中文分詞規範 1995
