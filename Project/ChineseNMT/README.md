# Chinese Neural Machine Translation

## Dataset

TED Talk (for translation corpus) - [readme](TED_data/README.txt)

* [The IWSLT 2015 Evaluation Campaign includes the MT track on TED Talks](https://wit3.fbk.eu/mt.php?release=2015-01)

## Usage

### Data proprocessing

```sh
# Data Preprocessing
mkdir output_vocab train_data
python3 DataPreprocessing.py
```

### Training

```sh
python3 train.py
```

Seq2seq

```txt
########################################################################
#      Date:           Tue Dec 25 17:34:15 PST 2018
#    Job ID:           8897.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=24:00:00
########################################################################

In iteration: 1
After 0 steps, per token cost is 8.295

...

After 9000 steps, per token cost is 2.397
After 9010 steps, per token cost is 2.525

########################################################################
# End of output for job 8897.c009
# Date: Wed Dec 26 00:23:54 PST 2018
########################################################################
```

Attention

```txt
########################################################################
#      Date:           Tue Dec 25 17:36:09 PST 2018
#    Job ID:           8900.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=24:00:00
########################################################################

In iteration: 1
After 0 steps, per token cost is 8.292

...

After 9000 steps, per token cost is 2.582
After 9010 steps, per token cost is 2.571

########################################################################
# End of output for job 8900.c009
# Date: Wed Dec 26 06:01:17 PST 2018
########################################################################
```

> Training attention model is much slower than seq2seq model (about two times)

### Test

```sh
python3 test.py
```

## Result

```txt
English: This is a test . <eos>
Chinese (seq2seq): <sos>这是一个测试。<eos>
Chinese (attention): <sos>这是一个测试。<eos>

English: Please give me a hundred on my final score . For the sake of how hard I paid on this course . <eos>
Chinese (seq2seq): <sos>請給我一分鐘。我的下一個問題是，我花了很長的時間才這樣做。<eos>
Chinese (attention): <sos>我最后一个人在这儿为我付出的钱。我的工作。<eos>
```

## TODO

Current version can only translate one sentence per run.
Make it a better way to reuse the model.

```txt
ValueError: Variable XXX already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
```

Multiple graph will conflict now.

## Links

* [Tensorflow Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
