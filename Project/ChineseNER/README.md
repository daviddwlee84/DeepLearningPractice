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

### Equivalent RNN Cell from scratch

### Use LSTM

## Links

* [Starting/stopping a background Python process wtihout nohup + ps aux grep + kill](https://stackoverflow.com/questions/34687883/starting-stopping-a-background-python-process-wtihout-nohup-ps-aux-grep-kill)