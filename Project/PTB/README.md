# Penn TreeBank

## Dataset

PTB - [readme](PTB_data/README)

* [RNNLM Toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/)
  * Download basic example `wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz`

TED Talk (for translation corpus) - [readme](TED_data/README.txt)

* [The IWSLT 2015 Evaluation Campaign includes the MT track on TED Talks](https://wit3.fbk.eu/mt.php?release=2015-01)

## Usage

```sh
# Data Preprocessing
mkdir output_vocab train_data
python3 DataPreprocessing.py
```

```sh
# Training Model
python3 PTBModel.py
```

### Result

```txt
########################################################################
#      Date:           Tue Dec 25 02:44:16 PST 2018
#    Job ID:           8783.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

In iteration: 1
After 0 steps, perplexity is 10003.722
After 100 steps, perplexity is 1730.202
After 200 steps, perplexity is 1173.143
After 300 steps, perplexity is 927.840
After 400 steps, perplexity is 766.663
After 500 steps, perplexity is 654.960
After 600 steps, perplexity is 580.428
After 700 steps, perplexity is 522.028
After 800 steps, perplexity is 470.585
After 900 steps, perplexity is 432.414
After 1000 steps, perplexity is 405.019
After 1100 steps, perplexity is 377.503
After 1200 steps, perplexity is 355.806
After 1300 steps, perplexity is 335.081
Epoch: 1 Train Perplexity: 331.853
Epoch: 1 Eval Perplexity: 184.656
In iteration: 2
After 1400 steps, perplexity is 180.056
After 1500 steps, perplexity is 165.544
After 1600 steps, perplexity is 167.702
After 1700 steps, perplexity is 164.732
After 1800 steps, perplexity is 160.043
After 1900 steps, perplexity is 157.864
After 2000 steps, perplexity is 156.240
After 2100 steps, perplexity is 151.399
After 2200 steps, perplexity is 148.324
After 2300 steps, perplexity is 146.970
After 2400 steps, perplexity is 144.628
After 2500 steps, perplexity is 141.715
After 2600 steps, perplexity is 138.285
Epoch: 2 Train Perplexity: 137.675
Epoch: 2 Eval Perplexity: 135.754
In iteration: 3
After 2700 steps, perplexity is 120.802
After 2800 steps, perplexity is 106.420
After 2900 steps, perplexity is 113.195
After 3000 steps, perplexity is 111.150
After 3100 steps, perplexity is 110.189
After 3200 steps, perplexity is 110.241
After 3300 steps, perplexity is 109.710
After 3400 steps, perplexity is 107.797
After 3500 steps, perplexity is 105.795
After 3600 steps, perplexity is 105.335
After 3700 steps, perplexity is 105.195
After 3800 steps, perplexity is 103.239
After 3900 steps, perplexity is 101.359
Epoch: 3 Train Perplexity: 100.962
Epoch: 3 Eval Perplexity: 116.750
In iteration: 4
After 4000 steps, perplexity is 99.655
After 4100 steps, perplexity is 85.016
After 4200 steps, perplexity is 90.511
After 4300 steps, perplexity is 90.439
After 4400 steps, perplexity is 89.596
After 4500 steps, perplexity is 89.158
After 4600 steps, perplexity is 88.950
After 4700 steps, perplexity is 88.136
After 4800 steps, perplexity is 86.719
After 4900 steps, perplexity is 86.211
After 5000 steps, perplexity is 86.418
After 5100 steps, perplexity is 85.046
After 5200 steps, perplexity is 84.094
After 5300 steps, perplexity is 83.636
Epoch: 4 Train Perplexity: 83.609
Epoch: 4 Eval Perplexity: 110.762
In iteration: 5
After 5400 steps, perplexity is 74.620
After 5500 steps, perplexity is 75.988
After 5600 steps, perplexity is 79.284
After 5700 steps, perplexity is 77.104
After 5800 steps, perplexity is 76.001
After 5900 steps, perplexity is 76.257
After 6000 steps, perplexity is 76.297
After 6100 steps, perplexity is 74.947
After 6200 steps, perplexity is 74.611
After 6300 steps, perplexity is 75.033
After 6400 steps, perplexity is 74.356
After 6500 steps, perplexity is 73.686
After 6600 steps, perplexity is 72.819
Epoch: 5 Train Perplexity: 72.972
Epoch: 5 Eval Perplexity: 107.839
Test Perplexity: 105.143

########################################################################
# End of output for job 8783.c009
# Date: Tue Dec 25 03:15:13 PST 2018
########################################################################
```

## Notes

perplexity

[Wiki - Perplexity](https://en.wikipedia.org/wiki/Perplexity)

## Links

* [TensorFlow PTB tutorial](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py)
  * [Stackoverflow](https://stackoverflow.com/questions/40786771/how-to-use-tensorflows-ptb-model-example)
* [Penn Tree Bank (PTB) dataset introduction](https://corochann.com/penn-tree-bank-ptb-dataset-introduction-1456.html)
  * [jupyter notebook](https://github.com/corochann/deep-learning-tutorial-with-chainer/blob/master/src/05_ptb_rnn/ptb/ptb_dataset_introduction.ipynb)
* [Penn Treebank tagset](https://www.sketchengine.eu/penn-treebank-tagset/)
