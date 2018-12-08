# Chinese Named Entity Recognition

[NER Notes](../../Notes/Application/NLP/NER.md)

## Dataset

* `<S>`: start of sentence
* `<E>`: end of sentence

Tag

* `O`: Normal word
* `B`: Begin of the Named Entity
* `I`: Body of the Named Entity

Detail

* `ORG`: Organization
* `PER`: Person
* `LOC`: Location

## Version

Change file `ner_backward.py` and `ner_test.py` to use different model.

```py
# ===== Use different model ===== #
import ner_forward_BasicRNNCell as ner_forward
#import ner_forward_fromScratch as ner_forward
#import ner_forward_LSTM as ner_forward
# =============================== #
```

Train model

```sh
# use nohup
nohup python3 -u ner_backward.py > log.txt 2>&1 &
# or use mynohup
./mynohup.sh ner_backward.py

# kill it
./mykill.sh ner_backward.py
```

See training logs

```sh
tail -f log.txt
```

Test (This will generate `./result/ner.result`)

```sh
python3 ner_test.py
```

### Use Tensorflow Basic RNN Cell

Training Result

```txt
########################################################################
#      Date:           Wed Dec  5 19:16:59 PST 2018
#    Job ID:           204772.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,vmem=92gb,walltime=06:00:00
########################################################################

...

After 95 training step(s), loss on training batch is 0.0798641.
After 96 training step(s), loss on training batch is 0.0783035.
After 97 training step(s), loss on training batch is 0.0778863.
After 98 training step(s), loss on training batch is 0.0776616.
After 99 training step(s), loss on training batch is 0.0764596.
After 100 training step(s), loss on training batch is 0.0763433.

########################################################################
# End of output for job 204772.c009
# Date: Wed Dec  5 19:39:44 PST 2018
########################################################################
```

Testing Result

```txt
After 100 training step(s), test result is:
              PER: precision:  43.32%; recall:  58.37%; FB1:  49.73
              LOC: precision:  37.49%; recall:  47.10%; FB1:  41.75
              ORG: precision:   8.06%; recall:  15.35%; FB1:  10.57
```

### Equivalent RNN Cell from scratch

Training Result

```txt
########################################################################
#      Date:           Thu Dec  6 01:53:35 PST 2018
#    Job ID:           204918.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,vmem=92gb,walltime=06:00:00
########################################################################

...

After 95 training step(s), loss on training batch is 0.0778476.
After 96 training step(s), loss on training batch is 0.0787005.
After 97 training step(s), loss on training batch is 0.0783031.
After 98 training step(s), loss on training batch is 0.0788957.
After 99 training step(s), loss on training batch is 0.0807808.
After 100 training step(s), loss on training batch is 0.0805757.

########################################################################
# End of output for job 204918.c009
# Date: Thu Dec  6 02:15:49 PST 2018
########################################################################
```

Testing Result

```txt
After 100 training step(s), test result is:
              PER: precision:  40.50%; recall:  55.20%; FB1:  46.72
              LOC: precision:  37.92%; recall:  51.00%; FB1:  43.50
              ORG: precision:   4.99%; recall:   9.76%; FB1:   6.60
```

### Use LSTM

It can simply just modify the "Key Part" in [ner_forward_BasicRNNCell.py](ner_forward_BasicRNNCell.py)

```py
### Key Part
nn_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
output, _ = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
###
```

Training Result

```txt
########################################################################
#      Date:           Sat Dec  8 01:25:38 PST 2018
#    Job ID:           205985.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,vmem=92gb,walltime=06:00:00
########################################################################

...

After 95 training step(s), loss on training batch is 0.00804608.
After 96 training step(s), loss on training batch is 0.00799358.
After 97 training step(s), loss on training batch is 0.00791584.
After 98 training step(s), loss on training batch is 0.00792954.
After 99 training step(s), loss on training batch is 0.00814747.
After 100 training step(s), loss on training batch is 0.00884923.

########################################################################
# End of output for job 205985.c009
# Date: Sat Dec  8 02:30:11 PST 2018
########################################################################
```

Testing Result

```txt
Got Serious bug...
```

I can completely trained to 100 steps using `python3 ner_backward.py`

But I can't use `python3 ner_test.py` to test. (Even though I can do it in the last two practice...). And I get the following error.

```txt
NotFoundError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key LSTMCell/Variable_1000 not found in checkpoint
         [[node save/RestoreV2 (defined at /Users/daviddwlee84/Documents/Program/PekingUniversity/DeepLearningPractice/Project/ChineseNER/ner_test.py:52)  = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save/Const_0_0, save/RestoreV2/tensor_names, save/RestoreV2/shape_and_slices)]]
```

Then I Googled some results and found a TensorFlow tool script - [**inspect_checkpoint.py**](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py)

And use it to check the tensors in the checkpoint.

```sh
python3 inspect_checkpoint.py --file_name=model/ner_model-100 > inspect.txt
```

And found that there is no such Key called LSTMCell/Variable_1000. But I still don't understand why it will need to look after this Key...

Maybe I'm not so familiar with TensorFlow's restoring mechanism...

So maybe I'll try it another time... this bug took me too much time...

## Links

* [Starting/stopping a background Python process wtihout nohup + ps aux grep + kill](https://stackoverflow.com/questions/34687883/starting-stopping-a-background-python-process-wtihout-nohup-ps-aux-grep-kill)
