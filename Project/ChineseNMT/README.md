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

### Test

```sh
python3 test.py
```

## Result

```txt
English: This is a test . <eos>
Chinese (seq2seq): <sos>這是一個測試。<eos>
Chinese (attention): <sos>這是一個測試。<eos>

English: Please give me a hundred on my final score . For the sake of how hard I paid on this course . <eos>
Chinese (seq2seq): <sos>請給我一分鐘。我的下一個問題是，我花了很長的時間才這樣做。<eos>
Chinese (attention): <sos>我最後一個人在這兒為我付出的錢。我的工作。<eos>

English: I just can't understand why I am so handsom . <eos>
Chinese (seq2seq): <sos>我只是想要解決我的問題。<eos>
Chinese (attention): <sos>我明白了為什麼我只是在理解這個理論，我為什麼只能在這個理論上解釋我的理解。<eos>
```

### Training time and loss

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

## TODO

### training

Current version can only translate one sentence per run.
Make it a better way to reuse the model.

```txt
ValueError: Variable XXX already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
```

### testing

Multiple graph will conflict now.

[TensorFlow Graphs and Sessions](https://www.tensorflow.org/guide/graphs) - Programming with multiple graphs

## Links

* [Tensorflow Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
