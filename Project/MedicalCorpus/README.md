# Medical Corpus Processing

## Overview

This experiment will include three parts of jobs.

1. Chinese word segmentation 詞語切分
2. Part-of-speech tagging 詞性標注
3. Named-entity recognition 命名實體識別

And with two phases

1. Using any tool, third-party corpus or even manual labelling the data set
2. Supervised Learning by using the given preprocessed data

### Data set / Corpus

* [raw_59.txt](data/raw_59.txt)
  * The raw data need to be tagged. Formed by random sample of source corpus.
  * Start with `sequence# + space`; End with `\n`.
* [context_59.txt](data/context_59.txt)
  * The ±3 lines of sentence surrounding the raw data sentence (if any).
  * Just for human validation.

### Corpus Processing Standard

#### Chinese word segmentation & POS tagging Standard

* [《北京大學現代漢語語料庫基本加工規範》](http://www.cinfo.net.cn/pz/pdf/a2%E5%8C%97%E4%BA%AC%E5%A4%A7%E5%AD%A6%E7%8E%B0%E4%BB%A3%E6%B1%89%E8%AF%AD%E8%AF%AD%E6%96%99%E5%BA%93%E5%9F%BA%E6%9C%AC%E5%8A%A0%E5%B7%A5%E8%A7%84%E8%8C%83.pdf)

#### Medical NER Standard

* 《醫學語料命名實體識別加工規範》

### Task

#### Chinese word segmentation

1. Must followed the standard. Each word segment must split by a single space.
2. Delete the meaningless space `$$_` (`\u0020`) or `$$__` (`\u3000`)

> the number of `_` means how many spaces are

Example of space

* Delete
  * `$$_` which is used to seperate number
    * e.g. `HCMV$$_150kD磷蛋白是HCMV蛋白结构中抗原性最强的蛋白`
* Don't Delete
  * `$$_` surrounding the `()`
    * e.g. `坏死性龈口炎$$_（necrotic$$_gingivostomatitis）`

#### Part-of-speech tagging (including General NER)

1. 26 Tags must followed the [Dictionary of Modern Chinese Grammar Information (現代漢語語法信息詞典)](http://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/EDQWIL)
2. The format must be `word/tag` (do not include space).
3. Additional General NER must include
   * nr: Name
   * ns: Place name
   * nt: Institution name

Confusing Example

Before|After
------|------
`“一、”、“（二）”、“3.”、“（4）”、“5）”`|`“一/m 、/w”、“（/w 二/m ）/w”、“3/m ./w”、“（/w 4/m ）/w”、“5/m ）/w”`
`abc<sub>xyz</sub>`|`abc<sub>xyz</sub>/n`

#### Medical NER

The format must be `[named-entity]tab`

Tag|NER
---|---
dis|disease
sym|symptom
tes|test
tre|treatment
bod|body part

Example

Before|After
------|------
`左下肺/n`|`[左下肺/n]bod`

### Evaluation

* Precision (P)
* Recall (R)
* F1-score (F1)
  * `F1 = 2 * P * R / (P + R)`

## First phase

### Chinese word segmentation by tool

### Part-of-speech tagging by tool

### Named-entity recognition by tool

## Second phase

### Chinese word segmentation by learning

### Part-of-speech tagging by learning

### Named-entity recognition by learning

## Resources

### NLP Tools

* [flair](https://github.com/zalandoresearch/flair) - A very simple framework for state-of-the-art NLP

#### Chinese

* [jieba](https://github.com/fxsjy/jieba) - 結巴中文分詞
* [pkuseg](https://github.com/lancopku/pkuseg-python)
  * [Trainer._decode_tokAcc](https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/trainer.py#L233) - token accuracy
* [THULAC](https://github.com/thunlp/THULAC-Python) (THU Lexical Analyzer for Chinese)
* [LTP](https://github.com/HIT-SCIR/ltp) (Language Technology Platform)
* [NLPIR](https://github.com/NLPIR-team/NLPIR)

### Article

* [中文分詞工具測評](https://rsarxiv.github.io/2016/11/29/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%B7%A5%E5%85%B7%E6%B5%8B%E8%AF%84/)
  * [SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)
    * [icwb2-data.zip](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip) - Score script (Evaluation), test gold data, training words data
