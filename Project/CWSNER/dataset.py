from constant import RAW_DATA, TRAIN_TEST
from typing import List, Tuple, Dict
import numpy as np
import io
import os
import pickle as pkl


def setup_cws_data(path: str = RAW_DATA.CWS):
    if not os.path.isfile(TRAIN_TEST.CWS_train) or not os.path.isfile(TRAIN_TEST.CWS_test):
        train_lines, test_lines = read_cws_data_and_split(path, 0.3)
        with open(TRAIN_TEST.CWS_train, 'w') as train_file:
            train_file.writelines(train_lines)
        with open(TRAIN_TEST.CWS_test, 'w') as test_file:
            test_file.writelines(test_lines)
    else:
        with open(TRAIN_TEST.CWS_train, 'r') as train_file:
            train_lines = train_file.readlines()
        with open(TRAIN_TEST.CWS_test, 'r') as test_file:
            test_lines = test_file.readlines()

    print("Word Segmentation Train and Test data split done.")

    if not os.path.isfile(TRAIN_TEST.CWS_test_raw):
        test_raw_list = get_raw_article_from_cws_data(
            TRAIN_TEST.CWS_test, TRAIN_TEST.CWS_test_raw)
    else:
        with open(TRAIN_TEST.CWS_test_raw, 'r') as test_raw:
            test_raw_list = test_raw.readlines()

    print("Word Segmentation Test raw article done. (to predict)")

    if not os.path.isfile(TRAIN_TEST.CWS_train_pkl):
        train_data_list = cws_transfer_to_trainable(train_lines)
        with open(TRAIN_TEST.CWS_train_pkl, 'wb') as train_pkl:
            pkl.dump(train_data_list, train_pkl)
    else:
        with open(TRAIN_TEST.CWS_train_pkl, 'rb') as train_pkl:
            train_data_list = pkl.load(train_pkl)

    print("Word Segmentation Trainable training data (70%) done.")

    if not os.path.isfile(TRAIN_TEST.CWS_test_pkl):
        test_data_list = cws_transfer_to_trainable(test_lines)
        with open(TRAIN_TEST.CWS_test_pkl, 'wb') as test_pkl:
            pkl.dump(test_data_list, test_pkl)
    else:
        with open(TRAIN_TEST.CWS_test_pkl, 'rb') as test_pkl:
            test_data_list = pkl.load(test_pkl)

    print("Word Segmentation Testable test data (30%) done.")

    if not os.path.isfile(TRAIN_TEST.CWS_train_all_pkl):
        raw_data_lines = load_utf16le_data_to_list(RAW_DATA.CWS)
        train_all_list = cws_transfer_to_trainable(raw_data_lines)
        with open(TRAIN_TEST.CWS_train_all_pkl, 'wb') as train_pkl:
            pkl.dump(train_all_list, train_pkl)
    else:
        with open(TRAIN_TEST.CWS_train_all_pkl, 'rb') as train_pkl:
            train_all_list = pkl.load(train_pkl)

    print("Word Segmentation Trainable training data (all) done.")

    return train_data_list, test_data_list, train_all_list, test_raw_list


def read_cws_data_and_split(path: str = RAW_DATA.CWS, test_ratio: float = 0.3):

    raw_data_lines = load_utf16le_data_to_list(path)

    total_lines = len(raw_data_lines)
    split_index = round(test_ratio * total_lines)

    np.random.shuffle(raw_data_lines)
    train_lines, test_lines = raw_data_lines[:-
                                             split_index], raw_data_lines[-split_index:]

    return train_lines, test_lines


def load_utf16le_data_to_list(path: str):
    """ load utf16-le data """
    with io.open(path, 'r', encoding='utf-16-le') as data_file:
        raw_data_lines = data_file.readlines()
    if len(raw_data_lines) <= 1:  # last line is empty line (only with '\n')
        del raw_data_lines[-1]
    return raw_data_lines


def cws_transfer_to_trainable(raw_data_lines: List[str]) -> List[List[Tuple[str, str]]]:
    """ Load labeled cws data into trainable format """

    dataset_list = []

    for raw_sentence in raw_data_lines:
        sentence_list = []
        for word in raw_sentence.split():
            if len(word) == 1:  # single word
                label = 'S'
                sentence_list.append((word, label))
            else:  # normal case
                for i, char in enumerate(word):
                    if i == 0:
                        label = 'B'
                    elif i == len(word)-1:
                        label = 'E'
                    else:
                        label = 'M'
                    sentence_list.append((char, label))

        dataset_list.append(sentence_list)

    return dataset_list


CWS_LabelEncode = {
    'B': 0,
    'M': 1,
    'E': 2,
    'S': 3
}

NER_LabelEncode = {
    'N': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6,
}


def get_total_word_set(train_all_dataset_list: list):
    """ statistic all the words in the training set and find the max sequence length and the word to id dict """
    all_word_in_train = [
        word for sentence in train_all_dataset_list for (word, _) in sentence]
    max_seq_len = max([len(sentence) for sentence in train_all_dataset_list])
    print('Max sentence (sequence) length:', max_seq_len)
    word_set = ['PAD'] + list(set(all_word_in_train))
    print('Total unique word (include PAD):', len(word_set))
    word_to_id = {word: index for index, word in enumerate(word_set)}

    return word_to_id, max_seq_len, word_set


def train_test_trainable_to_numpy(train_dataset_list: list, test_dataset_list: list, train_all_dataset_list: list, tag_to_id: Dict[str, int]):
    """ transfer all the training data into numpy array """
    word_to_id, max_seq_len, _ = get_total_word_set(
        train_all_dataset_list)

    train_x, train_y, train_seq_len = _single_trainable_to_numpy(
        train_dataset_list, word_to_id, tag_to_id, max_seq_len)
    test_x, test_y, test_seq_len = _single_trainable_to_numpy(
        test_dataset_list, word_to_id, tag_to_id, max_seq_len)
    all_x, all_y, all_seq_len = _single_trainable_to_numpy(
        train_all_dataset_list, word_to_id, tag_to_id, max_seq_len)

    return (train_x, train_y, train_seq_len), (test_x, test_y, test_seq_len), (all_x, all_y, all_seq_len)


def _single_trainable_to_numpy(dataset_list: List[List[Tuple[str, str]]], word_to_id: Dict[str, int], tag_to_id: Dict[str, int], max_seq_len: int):
    """ transfer single dataset into numpy array (word id vs. label) """
    x = []
    y = []
    seq_len = []
    for sentence in dataset_list:
        # transfer word to id and padding to max length
        x.append([word_to_id[word] if word in word_to_id else word_to_id['PAD'] for (word, _) in sentence] +
                 [word_to_id['PAD']] * (max_seq_len - len(sentence)))
        # padding the label with arbitrary label (these will be mask)
        y.append([tag_to_id[tag] for (_, tag) in sentence] +
                 [0] * (max_seq_len - len(sentence)))
        seq_len.append(len(sentence))

    return np.array(x), np.array(y), np.array(seq_len)


def get_raw_article_from_cws_data(path: str = TRAIN_TEST.CWS_test, output_path: str = ''):
    """ Transfer labeled cws data into raw article """
    with open(path, 'r') as f_in:
        raw_data_with_space = f_in.read()

    # remove all the space
    raw_article = "".join(raw_data_with_space.split(' '))

    if output_path:
        with open(output_path, 'w') as f_out:
            f_out.write(raw_article)

    return raw_article


if __name__ == "__main__":
    os.makedirs('train_test', exist_ok=True)
    train_data_list, test_data_list, train_all_list, test_raw_list = setup_cws_data()
    (train_x, train_y, train_seq_len), _, _ = train_test_trainable_to_numpy(
        train_data_list, test_data_list, train_all_list, CWS_LabelEncode)
    print(train_x)
