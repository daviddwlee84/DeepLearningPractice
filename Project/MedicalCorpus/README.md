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
2. Delete the meaningless space `$$_` (`\u0020`) or `$$__` (`\u3000`) (found that this should be doing befor segmentation)

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

| Before                                             | After                                                                             |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| `“一、”、“（二）”、“3.”、“（4）”、“5）”` | `“一/m 、/w”、“（/w 二/m ）/w”、“3/m ./w”、“（/w 4/m ）/w”、“5/m ）/w”` |
| `abc<sub>xyz</sub>`                                | `abc<sub>xyz</sub>/n`                                                             |

#### Medical NER

The format must be `[named-entity]tab`

| Tag | NER       |
| --- | --------- |
| dis | disease   |
| sym | symptom   |
| tes | test      |
| tre | treatment |
| bod | body part |

Example

| Before     | After           |
| ---------- | --------------- |
| `左下肺/n` | `[左下肺/n]bod` |

### Evaluation

> * N: gold segment words number
> * e: wrong number of word segment
> * c: correct number of word segment

* Precision (P) = c/N
* Recall (R) = c/(c+e)
* F1-score (F1)
  * `F1 = 2 * P * R / (P + R)`
* Error Rate (ER) = e/N (additional in this project)

## First phase

Idea:

1. clean up meaningless space (`$$_` first)
2. quick word segmentation using tool
3. make some rules to seperate words haven't been segmented or combine the mis-segmented words
4. ~~modify the POS table to fit the standard (26 tags) (e.g. `tag_to_idx` in pkuseg)~~. Map the POS to our standard.
5. doing medical dictionary on raw data and found the position of each medical NER
6. then decorate the previous result

Todo:

need to find the medical dictionary with tags to filter the medical named-entities

### Clean up meaningless space character

Check all the space (`$$_`) between words (no `$$__` exist in my raw data (e.g. line 33: `表6-15$$_$$_`))

> total number of `$$_` is 107 in my raw data.

```py
# observe the surrounding of some $$_
import re
with open('data/raw_59.txt') as f:
    text = f.read()
space_re = r'...\$\$_...'
re.findall(space_re, text)
```

```txt
['复循环$$_（C）', '3-1$$_第一年', '次0.$$_3g，', 'lus$$_Aci', 'lus$$_Cap', '菌0.$$_5亿，', '菌1.$$_35亿', '菌0.$$_15亿', '（5.$$_0～8', '/Qt$$_=（C', '2）×$$_100', ' 0.$$_5～1', '11.$$_5～1', '＞1.$$_020', '＞0.$$_009', '-15$$_$$_', 'p>/$$_L，嗜', '为2.$$_2kb', '为9.$$_9kb', 'tal$$_dia', '养治疗$$_此类患', '69.$$_4kJ', '日2.$$_29g', '第二节$$_生理性', '素0.$$_01～', '松0.$$_1～0', '-18$$_间隔缺', '∶10$$_000', '第一节$$_支气管', '＜6.$$_5kP', 'kPa$$_（60', '＜7.$$_20，', '-5.$$_0mm', '射0.$$_3～3', '次0.$$_5～1', 'mg=$$_125', '5U/$$_（kg', '素0.$$_5mg', '/kg$$_qd或', '于10$$_000', '或18$$_Gy（', 'Ron$$_T现象', 'ral$$_inf', 'ive$$_inf', 'ent$$_or$', 'ive$$_inf', 'ell$$_tra', 'low$$_vir', 'ian$$_stu', '_of$$_ren', 'ase$$_in$', '66.$$_1%为', '16.$$_1%为', '，8.$$_1%为', '为0.$$_5%～', '）头颅$$_MRI', '宿主病$$_（GV', 'hle$$_189', 'ial$$_hem', '为2.$$_5/1', '或0.$$_25%', 'mic$$_imp', '量0.$$_05～', ' 表2$$_常量和', '99.$$_9%。', 'ler$$_nod', '于5.$$_7mm', ' 1.$$_DIC', 'ked$$_AS，', '段Xq$$_22，', 'mal$$_rec', 'ive$$_AS，', 'mal$$_dom', 'ant$$_AS，', 'tal$$_hyp', '素试验$$_（结素', '症治疗$$_①静止', '泮0.$$_5mg', '第三节$$_肺结核', 'aan$$_vir', '、0.$$_5%碘', '（pH$$_3～5', ' 1.$$_ATP', 'APD$$_KT/', '为2.$$_0/w', '于1.$$_9/w', 'CPD$$_KT/', '为2.$$_1/w', 'IPD$$_KT/', '为2.$$_2/w', '低体温$$_体温常', 'ase$$_inh', 'ong$$_QT$', 'val$$_syn', '第四节$$_小儿药']
```

Delete the space (`$$_`) surrounding by number, decimal.

```py
replaced_space = re.sub(r'(\d.)\$\$_(\d)', r'\1\2', text)
```

Observe the rest of the spaces

```py
re.findall('....\$\$_....', replaced_space)
```

```txt
['表3-1$$_第一年小', 'llus$$_Acid', 'ilus$$_Caps', 's/Qt$$_=（Cc', 'O2）×$$_100%', '6-15$$_$$_S', 'up>/$$_L，嗜酸', 'atal$$_diag', '营养治疗$$_此类患者', ' 第二节$$_生理性贫', '9-18$$_间隔缺损', ' 第一节$$_支气管哮', '8kPa$$_（60m', '1mg=$$_125U', '15U/$$_（kg•', 'g/kg$$_qd或b', '）或18$$_Gy（年', '④Ron$$_T现象；', 'iral$$_infe', 'tive$$_infe', 'tent$$_or$$', 'tive$$_infe', 'cell$$_tran', 'slow$$_viru', 'sian$$_stud', '$_of$$_rena', 'ease$$_in$$', '抗宿主病$$_（GVH', 'ehle$$_1897', 'nial$$_hemo', 'omic$$_impr', '8 表2$$_常量和微', 'sler$$_node', '6 1.$$_DIC治', 'nked$$_AS，X', '中段Xq$$_22，为', 'omal$$_rece', 'sive$$_AS，A', 'omal$$_domi', 'nant$$_AS，A', 'ital$$_hypo', '菌素试验$$_（结素试', '对症治疗$$_①静止性', ' 第三节$$_肺结核病', 'taan$$_viru', '酸（pH$$_3～5）', '6 1.$$_ATP耗', 'CAPD$$_KT/V', 'CCPD$$_KT/$', 'NIPD$$_KT/V', '.低体温$$_体温常在', 'rase$$_inhi', 'long$$_QT$$', 'rval$$_synd', ' 第四节$$_小儿药物']
```

Maybe we should leave the rest of the things

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

* **Evaluation of the default performance of segmentation**
  * jieba (jieba has auto `'\n'` problem. So this report is not quite fair)

    ```txt
    === Evaluation reault of word segment ===
    F1: 60.61%
    P : 60.87%
    R : 60.34%
    ER: 40.00%
    =========================================
    ```

  * pkuseg

    ```txt
    === Evaluation reault of word segment ===
    F1: 83.11%
    P : 79.13%
    R : 87.50%
    ER: 11.30%
    =========================================
    ```

Original setting segmentation problem

* `'应详细'`
  * jieba: `'应', '详细'` (O)
  * pkuseg: `'应详细'`
* `'三凹征'`
  * jieba: `'三', '凹征'`
  * pkuseg: `'三凹征'` (O)
* `表3-1`
  * jieba: `表 3 - 1`
  * pkuseg: `表 3&1`
  * [Dealing with number-number](#Dealing-with-number-number) (7 `\d-\d` pattern)

**After solving `$$_` and auto `\n` problem**:

* jieba

    ```txt
    === Evaluation reault of word segment ===
    F1: 88.11%
    P : 86.96%
    R : 89.29%
    ER: 10.43%
    =========================================
    ```

* pkuseg

    ```txt
    === Evaluation reault of word segment ===
    F1: 85.71%
    P : 80.87%
    R : 91.18%
    ER: 7.83%
    =========================================
    ```

#### Soluiton for customized segment

jieba (the `user_dict_file` [example](https://github.com/fxsjy/jieba/blob/master/test/test_userdict.py))

```py
jieba.load_userdict(user_dict_file_name)
jieba.add_word(word, freq=None, tag=None)
jieba.suggest_freq(segment, tune=True)
```

pkuseg

```py
pkuseg.pkuseg(user_dict='my_dict.txt')
```

> [POS User dictionary](#User-dictionary)

#### Last name problem

1. Get the last name list on the internet.
2. Split the word length grater than "a last name" with `/nr` tag.

Here is the imporvement after split name.

```txt
Test jieba word segmentation
=== Evaluation reault of word segment ===
F1: 100.00%
P : 100.00%
R : 100.00%
ER: 0.00%
=========================================
```

```txt
Test pkuseg word segmentation
line: 1 found error: (5, 8) => 应详细
line: 1 found error: (40, 46) => 自主心跳呼吸
line: 1 found error: (70, 73) => 光反应
line: 3 found error: (3, 6) => 缺损者
line: 3 found error: (21, 26) => 短暂菌血症
line: 3 found error: (32, 35) => 创伤性
line: 3 found error: (43, 46) => 细菌性
line: 4 found error: (3, 10) => 耀辉$$_孙锟
=== Evaluation reault of word segment ===
F1: 87.56%
P : 83.33%
R : 92.23%
ER: 7.02%
=========================================
```

wierd thing tagging with nr

```txt
龙 nr
粟粒状 nr
阿托品 nr
埃希菌 nr
克雷伯 nr
广谱抗 nr
维生素 nr
青少年 nr
晨 nr
左心室 nr
毛发 nr
内含子 nr
甘露醇 nr
张力 nr
帕米来 nr
律 nr
段 nr
过敏 nr
雷诺 nr
周 nr
洛贝林 nr
安全性 nr
凯瑞 nr
青光眼 nr
应予以 nr
常继发 nr
门静脉 nr
史 nr
幸存者 nr
高达 nr
地高辛 nr
关键因素 nr
小梁 nr
束 nr
迟发性 nr
地西泮 nr
巨 nr
欧氏 nr
张力 nr
白蛋白 nr
若有阳 nr
显微镜 nr
巧克力 nr
灵敏性 nr
麻醉 nr
利培 nr
麻风 nr
马拉 nr
姬鼠 nr
高峰 nr
易 nr
青壮年 nr
行为矫正 nr
青少年 nr
广谱抗 nr
```

#### The `<sup></sup>` and `<sub></sub>` problem

* `<sup></sup>`
  * `10<sup>12</sup>`
  * `Ca<sup>2+</sup>`
  * `10<sup>9</sup>`
  * `10<sup>9</sup>`
  * `<sup>*</sup>`
* `<sub></sub>`
  * `PaO<sub>2</sub>`
  * `CO<sub>2</sub>`
  * `U<sub>1</sub>`
  * `PaO<sub>2</sub>`
  * `PaCO<sub>2</sub>`
  * `PaO<sub>2</sub>`
  * `PaCO<sub>2</sub>`
  * `CD<sub>33</sub>`
  * `CD<sub>13</sub>`
  * `CD<sub>15</sub>`
  * `CD<sub>11</sub>b`
  * `CD<sub>36</sub>`

### Part-of-speech tagging by tool

* [pkuseg POS](#pkuseg-POS)
* [jieba POS](#jieba-POS)

#### Dealing with number-number

pkuseg: `表3-1` -> `表/n 3&1/v`

Find all the pos string with & in it.

#### User dictionary

```py
from pkuseg import pkuseg
pseg = pkuseg(model_name='medicine', postag=True,
              user_dict='user_dict/user_dict.txt')
jieba.load_userdict('user_dict/user_dict.txt')
import jieba.posseg as jseg
```

jieba dictionary format ([example](https://github.com/fxsjy/jieba/blob/master/test/userdict.txt))

`word`, `word_frequency(optional)`, `pos_tag(optional)`

> pkuseg only have `word`...

Effect after using user dictionary

* `三凹征 fuckyou`
  * jieba: `'三凹征/fuckyou'`
  * pkuseg: `'三凹征/j'`

### Named-entity recognition by tool

> (deprecated) Using the medicine corpus offered by pkuseg ([release v0.0.16](https://github.com/lancopku/pkuseg-python/releases/tag/v0.0.16))
>
> This contain a string with medical words seperated by `\n` (but also other words...)

Found some thing in previous result which need to be fixed.

* `小儿脑性/n 瘫痪/v`

Idea: Common pattern

* `XX症`
* `XX炎`
* `XX 損傷`

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

* [[原創]中文分詞器分詞效果的評測方法](https://www.codelast.com/%E5%8E%9F%E5%88%9B%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%99%A8%E5%88%86%E8%AF%8D%E6%95%88%E6%9E%9C%E7%9A%84%E8%AF%84%E6%B5%8B%E6%96%B9%E6%B3%95/)
* [中文分詞工具測評](https://rsarxiv.github.io/2016/11/29/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%B7%A5%E5%85%B7%E6%B5%8B%E8%AF%84/)
  * [SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)
    * [icwb2-data.zip](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip) - Score script (Evaluation), test gold data, training words data

### Regular Expression

* [**Stackoverflow - Python string.replace regular expression**](https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression)
* [w3schools Python RegEx](https://www.w3schools.com/python/python_regex.asp)
* [txt2re](http://txt2re.com/)
* [Regular-Expressions.info](https://www.regular-expressions.info/)

## Other

### TODO

* [X] fix `$$_`
* [X] add user dictionary
  * [ ] symbol and english didn't work
* [ ] [num-num problem](#Dealing-with-number-number)
* [ ] `<sup> </sup>` `<sub> </sub>`
* [ ] medical NER
  * [ ] find corpus
    * [A 醫學百科 症狀](http://www.a-hospital.com/w/%E7%97%87%E7%8A%B6)
  * [ ] split
  * [ ] add tag

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

#### dictionary format

medicine corpus

`.pkuseg/medicine/features.pkl` => a `dict`

```py
import pickle as pkl
features = pkl.load(open('features.pkl', 'rb'))

features = {
    'unigram': ...,
    'bigram': ...,
    'feature_to_idx': ...,
    'tag_to_idx': ...
}
```

`.pkuseg/medicine/medicine_dict.pkl` => a `str`

```py
medicine = pkl.load(open('medicine_dict.pkl', 'rb'))
medicine_dict = medicine.split('\n')
```

### jieba trace code

#### jieba append dictionary

> TODO: maybe try to use the dictionary offered by pkuseg for jieba (maybe need some adjustment)

Get the medical dictionary from pkuseg. Subtract the general words in other general corpus/dictionary. Then insert into jeiba

```py
medicine = pickle.load(open('%smedicine_dict.pkl' % pickle_dir, 'rb'))
medicine_dict = medicine.split('\n')

ctb8 = pickle.load(open('%sctb8.pkl' % pickle_dir, 'rb'))
msra = pickle.load(open('%smsra.pkl' % pickle_dir, 'rb'))
weibo = pickle.load(open('%sweibo.pkl' % pickle_dir, 'rb'))
other_dict = set([*ctb8, *msra, *weibo])
```

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

### Generator

* TypeError: object of type 'generator' has no len()
* TypeError: 'generator' object is not subscriptable

### Deprecated notes

#### Clean up space character

```py
re.findall(r'(\D\D)\$\$_(\D\D)', text)
```

```txt
[('循环', '（C'), ('us', 'Ac'), ('us', 'Ca'), ('Qt', '=（'), ('$_', 'SI'), ('>/', 'L，'), ('al', 'di'), ('治疗', '此类'), ('二节', '生理'), ('一节', '支气'), ('U/', '（k'), ('kg', 'qd'), ('on', 'T现'), ('al', 'in'), ('ve', 'in'), ('nt', 'or'), ('ve', 'in'), ('ll', 'tr'), ('ow', 'vi'), ('us', 'in'), ('an', 'st'), ('dy', 'of'), ('al', 'di'), ('se', 'in'), ('头颅', 'MR'), ('主病', '（G'), ('al', 'he'), ('ic', 'im'), ('er', 'no'), ('ed', 'AS'), ('al', 're'), ('ve', 'AS'), ('al', 'do'), ('nt', 'AS'), ('al', 'hy'), ('试验', '（结'), ('治疗', '①静'), ('三节', '肺结'), ('an', 'vi'), ('PD', 'KT'), ('PD', 'KT'), ('PD', 'KT'), ('体温', '体温'), ('se', 'in'), ('ng', 'QT'), ('al', 'sy'), ('四节', '小儿')]
```

Maybe add a rule. If a space (`$$_`) surrounding by numbers in 2~3 letter. Then delete it. Otherwise, ~~replace it with normal space.~~ keep it.

> * `string.replace("pattern", "replace")`
> * `re.sub(r"pattern", "replace", string)`

```py
# Replace all the '$$_' with ' '
# all the $$_ not surrounding by digital
replaced_space = re.sub(r'(\D\D)\$\$_(\D\D)', r'\1 \2', text)
# all the $$_ surrounding by english letter
replaced_space = re.sub(r'(\w)\$\$_(\w)', r'\1 \2', replaced_space)

# check the rest of the '$$_'
re.findall(space_re, replaced_space)
```

```txt
['次0.$$_3g，', '菌0.$$_5亿，', '菌1.$$_35亿', '菌0.$$_15亿', '（5.$$_0～8', '2）×$$_100', ' 0.$$_5～1', '11.$$_5～1', '＞1.$$_020', '＞0.$$_009', '-15$$_ SI', '为2.$$_2kb', '为9.$$_9kb', '69.$$_4kJ', '日2.$$_29g', '素0.$$_01～', '松0.$$_1～0', '＜6.$$_5kP', 'kPa$$_（60', '＜7.$$_20，', '-5.$$_0mm', '射0.$$_3～3', '次0.$$_5～1', 'mg=$$_125', '素0.$$_5mg', '66.$$_1%为', '16.$$_1%为', '，8.$$_1%为', '为0.$$_5%～', '为2.$$_5/1', '或0.$$_25%', '量0.$$_05～', '99.$$_9%。', '于5.$$_7mm', ' 1.$$_DIC', '泮0.$$_5mg', '、0.$$_5%碘', ' 1.$$_ATP', '为2.$$_0/w', '于1.$$_9/w', 'KT/$$_Vur', '为2.$$_1/w', '为2.$$_2/w']
```

> There is a weird thing `KT/$$_Vur` => On the 186th line `CCPD$$_KT/$$_Vurea为2.$$_1/w，NIPD$$_KT/Vurea为` => There are `KT/Vurea` following it. So it should delete as usual.

```py
# Delete all the other '$$_'
cleaned_text = replaced_space.replace('$$_', '')
```

#### jieba space combiner

```py
# Combine '$', '$', '_' into $$_
# Combine '$/x', '$/x', '_/x' into $$_
# equivalent to join than replace '$ $ _ ' (delete)
def _jiebaSpaceCombiner(pos_or_word_seg_list:list, func:str):
    assert func in ('seg', 'pos')
    if func == 'pos':
        print(pos_or_word_seg_list)
    idx_head_to_pop_from_list = []
    for i in range(len(pos_or_word_seg_list)-2):
        if func == 'seg' and pos_or_word_seg_list[i:i+3] == ['$', '$', '_']:  # for word segment
            idx_head_to_pop_from_list.append(i)
        elif func == 'pos' and pos_or_word_seg_list[i:i+3] == ['$/x', '$/x', '_/x']:  # for POS
            idx_head_to_pop_from_list.append(i)
    shift = 0
    for j in idx_head_to_pop_from_list:
        j += shift
        for k in range(3):
            pos_or_word_seg_list.pop(j)
            if k == 2:
                pos_or_word_seg_list.insert(j, '$$_')
                shift -= 2
    return pos_or_word_seg_list



# in _firstWordSegmentationWithPOS
if tools == 'jieba':
    word_seg_list_dict[seq_num] = _jiebaSpaceCombiner(word_seg_list_dict[seq_num], func='seg')
    pre_pos_list_dict[seq_num] = _jiebaSpaceCombiner(pre_pos_list_dict[seq_num], func='pos')
```
