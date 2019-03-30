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

> The output filename must be `1_ 59` and `2_ 59` for two phases result. (just inlcude the result after Medical NER)

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

Idea:

1. quick word segmentation using tool
2. make some rules to seperate words haven't been segmented or combine the mis-segmented words
3. modify the POS table to fit the standard (26 tags) (e.g. `tag_to_idx` in pkuseg)

Todo:

need to find the medical dictionary with tags to filter the medical named-entities

### Chinese word segmentation by tool

Tried pkuseg with medicine model and jieba

```py
# Word segmentation with POS tagging

# pkuseg
from pkuseg import pkuseg
pseg = pkuseg(model_name='medicine', postag=True)
words = pseg.cut(chinese_string)

# jieba
import jieba.posseg as jseg
words = jseg.cut(chinese_string)

for word, flag in words:
    pass
```

### Part-of-speech tagging by tool

### Named-entity recognition by tool

> (deprecated) Using the medicine corpus offered by pkuseg ([release v0.0.16](https://github.com/lancopku/pkuseg-python/releases/tag/v0.0.16))
>
> This contain a string with medical words seperated by `\n` (but also other words...)

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
* [THULAC](https://github.com/thunlp/THULAC-Python) (THU Lexical Analyzer for Chinese)
* [LTP](https://github.com/HIT-SCIR/ltp) (Language Technology Platform)
* [NLPIR](https://github.com/NLPIR-team/NLPIR)

### Article

* [中文分詞工具測評](https://rsarxiv.github.io/2016/11/29/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%B7%A5%E5%85%B7%E6%B5%8B%E8%AF%84/)
  * [SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)
    * [icwb2-data.zip](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip) - Score script (Evaluation), test gold data, training words data

## Other

### pkuseg trace code

#### token accuracy

* [Trainer._decode_tokAcc](https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/trainer.py#L233) - token accuracy

#### download model

> default location `~/.pkuseg`

* [download_model](https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/download.py#L30) - called in [`__init__.py`](https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/__init__.py#L183)
  * [model_urls](https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/config.py#L20)

#### pkuseg POS

POS Tags: [`tags.txt`](https://github.com/lancopku/pkuseg-python/blob/master/tags.txt)

dict: `tag_to_idx`

There are 35 different tags. But in our standard we only have 26. Thus we need some sort of map.

And I found that the first 26 POS is match the standard. The pkuseg has done some extra work on NER.

```txt
nr  人名
ns  地名
nt  机构名称
nx  外文字符
nz  其它专名
vd  副动词
vn  名动词
vx  形式动词
ad  副形词
an  名形词
```

But we only need `nr`, `ns` and `nt` in this experiment.

So I map `nx`, `nz` to `n`. And map `vd`, `vn`, `vx` to `v`. And map `ad`, `an` to `a`

### jieba trace code

#### jieba POS

* [GithubGist - hscspring/结巴词性标记集](https://gist.github.com/hscspring/c985355e0814f01437eaf8fd55fd7998)

#### medicine dictionary

> string

```py
In [15]: a[:100]
Out[15]: '中国\n发展\n工作\n经济\n国家\n记者\n我们\n一个\n问题\n建设\n人民\n全国\n进行\n政府\n社会\n市场\n他们\n改革\n下\n北京\n我国\n国际\n地区\n管理\n领导\n公司\n技术\n关系\n世界\n重要\n干部\n美国\n组织\n群众'

In [17]: a[20000:20100]
Out[17]: '\n有的是\n服务器\n味精\n男生\n行当\n咀嚼\n博爱\n丛林\n和平区\n冒充\n小国\n滨州\n逆向\n漏水\n咽喉\n潜伏\n潜水\n中信\n灵芝\n天涯\n中年人\n白人\n自备\n触摸\n俗称\n刘建国\n诊疗\n反倒\n改动\n说说\n节制\n板'
```
