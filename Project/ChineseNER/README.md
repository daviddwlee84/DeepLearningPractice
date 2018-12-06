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

## Links

* [Starting/stopping a background Python process wtihout nohup + ps aux grep + kill](https://stackoverflow.com/questions/34687883/starting-stopping-a-background-python-process-wtihout-nohup-ps-aux-grep-kill)
