# https://www.kaggle.com/thousandvoices/simple-lstm
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import text, sequence
from keras.optimizers import Adam
from keras.layers import LSTM, CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.models import Model
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas()

USE_GPU = False

if USE_GPU:
    from keras.layers import CuDNNLSTM
else:
    from keras.layers import LSTM

# crawl-300d-2M.vec--> https://fasttext.cc/docs/en/english-vectors.html
# When pre-train embedding is helpful? https://www.aclweb.org/anthology/N18-2084
# There are many pretrained word embedding models:
# fasttext, GloVe, Word2Vec, etc
# crawl-300d-2M.vec is trained from Common Crawl (a website that collects almost everything)
# it has 2 million words. Each word is represent by a vector of 300 dimensions.

# https://nlp.stanford.edu/projects/glove/
# GloVe is similar to crawl-300d-2M.vec. Probably, they use different algorithms.
# glove.840B.300d.zip: Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
# tokens mean words. It has 2.2M different words and 840B (likely duplicated) words in total

# note that these two pre-trained models give 300d vectors.
EMBEDDING_FILES = [
    'embedding/crawl-300d-2M.vec',
    'embedding/glove.840B.300d.txt'
]

EMBEDDING_URL = [
    'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip',
    'http://nlp.stanford.edu/data/glove.840B.300d.zip'
]

# check if the files exist
# make sure the "embedding" folder exist
os.makedirs('embedding', exist_ok=True)
for i, f in enumerate(EMBEDDING_FILES):
    if not os.path.isfile(f):
        print(os.path.basename(EMBEDDING_URL[i]), 'not found. Downloading...')
        os.system('wget ' + EMBEDDING_URL[i])
        os.system('unzip ' + os.path.basename(EMBEDDING_URL[i]))
        os.system('mv ' + os.path.basename(f) + ' embedding')
        os.system('rm ' + os.path.basename(EMBEDDING_URL[i]))

NUM_MODELS = 2
# the maximum number of different words to keep in the original texts
# 40_000 is a normal number
# 100_000 seems good too
MAX_FEATURES = 100000

# this is the number of training sample to put in theo model each step
BATCH_SIZE = 512

# units parameters in Keras.layers.LSTM/CuDNNLSTM
# it it the dimension of the output vector of each LSTM cell.
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

# we will convert each word in a comment_text to a number.
# So a comment_text is a list of number. How many numbers in this list?
# we want the length of this list is a constant -> MAX_LEN
MAX_LEN = 220


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    # each line in the file looks like
    # apple 0.3 0.4 0.5 0.6 ...
    # that is a word followed by 300 float numbers

    with open(path) as f:
        # return dict(get_coefs(*line.strip().split(' ')) for line in f)
        return dict(get_coefs(*o.strip().split(" ")) for o in tqdm(f))


def build_matrix(word_index, path):
    # path: a path that contains embedding matrix
    # word_index is a dict of the form ('apple': 123, 'banana': 349, etc)
    # that means word_index[word] gives the index of the word
    # word_index was built from all commment_texts

    # we will construct an embedding_matrix for the words in word_index
    # using pre-trained embedding word vectors from 'path'

    embedding_index = load_embeddings(path)

    # embedding_matrix is a matrix of len(word_index)+1  x 300
    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    # word_index is a dict. Each element is (word:i) where i is the index
    # of the word
    for word, i in word_index.items():
        try:
            # RHS is a vector of 300d
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix


def build_model(embedding_matrix, num_aux_targets):
   # a simpler version can be found here
   # https://www.tensorflow.org/tutorials/keras/basic_text_classification

   # Trainable params of the model: 1,671,687
   # Recall that the number of samples in train.csv is 1_804_874

    # words is a vector of MAX_LEN dimension
    words = Input(shape=(MAX_LEN,))

    # Embedding is the keras layer. We use the pre-trained embbeding_matrix
    # https://keras.io/layers/embeddings/
    # have to say that parameters in this layer are not trainable
    # x is a vector of 600 dimension
    x = Embedding(*embedding_matrix.shape,
                  weights=[embedding_matrix], trainable=False)(words)

    # *embedding_matrix.shape is a short way for
    #input_dim = embedding_matrix.shape[0], output_dim  = embedding_matrix.shape[1]

    # here the author used pre-train embedding matrix.
    # instead of train from begining like in tensorflow example

    # https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
    x = SpatialDropout1D(0.25)(x)

    if USE_GPU:
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    else:
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])

    hidden = add(
        [hidden, Dense(DENSE_HIDDEN_UNITS, activation='tanh')(hidden)])
    hidden = add(
        [hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid', name='main_output')(hidden)

    # num_aux_targets = 6 since y_aux_train has 6 columns
    aux_result = Dense(num_aux_targets, activation='sigmoid',
                       name='aux_ouput')(hidden)

    model = Model(inputs=words, outputs=[result, aux_result])

    # model.summary() will gives a good view of the model structure

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(clipnorm=0.1),
        metrics=['accuracy'])

    return model

DATAFOLDER = 'data'

DATASET = {
    'train': DATAFOLDER + '/train.csv',
    'test': DATAFOLDER + '/test.csv',
}

# check if the files exist
os.makedirs(DATAFOLDER, exist_ok=True)  # make sure the "data" folder exist
if not os.path.isfile(DATASET['train']) or not os.path.isfile(DATASET['test']):
    print('Dataset not found. Downloading from Kaggle...')
    os.system(
        'kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification')
    for f in DATASET.values():
        os.system('unzip ' + os.path.basename(f) + '.zip -d ' + DATAFOLDER)
    os.system('chmod +r ' + DATAFOLDER + '/*')
    os.system('rm *.zip')

train = pd.read_csv(DATASET['train'])
test = pd.read_csv(DATASET['test'])

# Take the columns 'comment_text' from train,
# then fillall NaN values by emtpy string '' (redundant)
x_train = train['comment_text'].fillna('').values

# if true, y_train[i] =1, if false, it is = 0
y_train = np.where(train['target'] >= 0.5, 1, 0)

y_aux_train = train[['target', 'severe_toxicity',
                     'obscene', 'identity_attack', 'insult', 'threat']]

# Take the columns 'comment_text' from test,
# then fillall NaN values by emtpy string '' (redundant)
x_test = test['comment_text'].fillna('').values

# https://keras.io/preprocessing/text/
# tokenizer is a class with some method
tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

# we apply method fit_on_texts of tokenizer on x_train and x_test
# it will initialize some parameters/attribute inside tokenizer
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L139
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L210

tokenizer.fit_on_texts(list(x_train) + list(x_test))
# for example, after fit_on_texts, we can type
# tokenizer.word_counts #give a OderedDict
# tokenizer.document_counts # an int
# tokenizer.word_index is a dict of words with correponding indices
# There are 410046 different words in all 'comment_text'
#len(tokenizer.word_index) == 410_046


# these words come from all 'comment_text' in training.csv and test.csv

# tokenizer.index_word: the inverse of tokenizer.word_index


# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L267
# we will convert each word in a comment_text to a number.
# So a comment_text is a list of number.


x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# https://keras.io/preprocessing/sequence/
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py
# each comment_text is now a list of word
# we want the length of this list is a constant -> MAX_LEN
# if the list is longer, then we cut/trim it
# if shorter, then we add/pad it with 0's at the beginning
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)


# create an embedding_matrix
# after this, embedding_matrix is a matrix of size
# len(tokenizer.word_index)+1   x 600
# we concatenate two matrices, 600 = 300+300
embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
# embedding_matrix.shape
# == (410047, 600)

# embedding_matrix[i] is a 600d vector representation of the word whose index is i
# embedding_matrix[10]
#tokenizer.index_word[10] == 'you'


checkpoint_predictions = []
weights = []


# https://keras.io/callbacks/#learningratescheduler

for model_idx in range(NUM_MODELS):
    # build the same models
    model = build_model(embedding_matrix, y_aux_train.shape[-1])
    # We train each model EPOCHS times
    # After each epoch, we reset learning rate (we are using Adam Optimizer)
    # https://towardsdatascience.com/learning-rate-scheduler-d8a55747dd90

    # https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L921
    # learningrate is the attribute 'lr' from Adam optimizer
    # see https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L460
    # In Adam Optimizer, learning rate is changing after each batch
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            callbacks=[
                LearningRateScheduler(
                    lambda epoch: 1e-3 * (0.6 ** global_epoch), verbose=1)
            ]
        )
        # model.predict will give two outputs: main_output (target) and aux_output
        # we only take main_output
        checkpoint_predictions.append(model.predict(
            x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)


# take average (with weights) of predictions from two models
# predictions is an np.array
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False)
