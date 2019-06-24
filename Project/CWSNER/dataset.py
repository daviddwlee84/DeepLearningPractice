from constant import RAW_DATA, TRAIN_TEST
from typing import List, Tuple, Dict
import numpy as np
import io
import os
import pickle as pkl

# CWS Function


def setup_cws_data():
    if not os.path.isfile(TRAIN_TEST.CWS_train) or not os.path.isfile(TRAIN_TEST.CWS_test):
        train_lines, test_lines = read_cws_data_and_split(RAW_DATA.CWS, 0.3)
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

    if not os.path.isfile(TRAIN_TEST.CWS_final_pkl):
        final_raw_lines = load_utf16le_data_to_list(RAW_DATA.CWS_test)
        final_raw_list = [line.strip() for line in final_raw_lines]
        with open(TRAIN_TEST.CWS_final_pkl, 'wb') as final_pkl:
            pkl.dump(final_raw_list, final_pkl)
    else:
        with open(TRAIN_TEST.CWS_final_pkl, 'rb') as final_pkl:
            final_raw_list = pkl.load(final_pkl)

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

    return train_data_list, test_data_list, train_all_list, final_raw_list


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


def from_trainable_to_cws_list(dataset_list: List[List[Tuple[str, str]]], output_path: str = ''):
    cws_list = []
    for sentence in dataset_list:
        str_sentence = ""
        for i, (word, label) in enumerate(sentence):
            if i < len(sentence) and (label == 'S' or label == 'E'):
                str_sentence += word + '  '  # in original data it use two spaces to seperate words
            elif i == len(sentence) or label == 'B' or label == 'M':
                str_sentence += word
        cws_list.append(str_sentence)

    if output_path:
        with open(output_path, 'w') as f_out:
            for str_sentence in cws_list:
                f_out.write(str_sentence + '\n')

    return cws_list


# Deprecated: can't deal with OOV word
def from_cws_numpy_to_evaluable_format(x, y, seq_len, word_to_id: Dict[str, int], tag_to_id: Dict[str, int] = CWS_LabelEncode, output_path: str = ''):
    all_sentences = from_numpy_to_trainable(
        x, y, seq_len, word_to_id, tag_to_id)
    return from_trainable_to_cws_list(all_sentences, output_path)

def combine_cws_numpy_pred_to_evaluable_format(dataset_list, pred, tag_to_id: Dict[str, int] = CWS_LabelEncode, output_path: str = ''):
    all_sentences = combine_numpy_pred_to_trainable(dataset_list, pred, tag_to_id)
    return from_trainable_to_cws_list(all_sentences, output_path)

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


# NER Function

NER_LabelEncode = {
    'N': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6,
}


def setup_ner_data():

    if not os.path.isfile(TRAIN_TEST.NER_train) or not os.path.isfile(TRAIN_TEST.NER_test):
        train_data_list, test_data_list = read_ner_data_and_split(
            RAW_DATA.NER, 0.3, use_utf16_encoding=True)
        from_trainable_to_ner_list(train_data_list, TRAIN_TEST.NER_train)
        from_trainable_to_ner_list(test_data_list, TRAIN_TEST.NER_test)
    else:
        train_data_list = _read_ner_data_to_trainable(
            TRAIN_TEST.NER_train, has_label=True, use_utf16_encoding=False)
        test_data_list = _read_ner_data_to_trainable(
            TRAIN_TEST.NER_test, has_label=True, use_utf16_encoding=False)

    print("Named Entity Recognition Train and Test data split done.")
    print("Named Entity Recognition Trainable training data (70%) done.")
    print("Named Entity Recognition Testable test data (30%) done.")

    if not os.path.isfile(TRAIN_TEST.NER_final_pkl):
        final_raw_list = _read_ner_data_to_trainable(
            RAW_DATA.NER_test, has_label=False, use_utf16_encoding=True)
        with open(TRAIN_TEST.NER_final_pkl, 'wb') as final_pkl:
            pkl.dump(final_raw_list, final_pkl)
    else:
        with open(TRAIN_TEST.NER_final_pkl, 'rb') as final_pkl:
            final_raw_list = pkl.load(final_pkl)

    print("Named Entity Recognition Test raw article done. (to predict)")

    if not os.path.isfile(TRAIN_TEST.NER_train_all_pkl):
        train_all_data_list = _read_ner_data_to_trainable(
            RAW_DATA.NER, has_label=True, use_utf16_encoding=True)
        with open(TRAIN_TEST.NER_train_all_pkl, 'wb') as train_pkl:
            pkl.dump(train_all_data_list, train_pkl)
    else:
        with open(TRAIN_TEST.NER_train_all_pkl, 'rb') as train_pkl:
            train_all_data_list = pkl.load(train_pkl)

    print("Named Entity Recognition Trainable training data (all) done.")

    return train_data_list, test_data_list, train_all_data_list, final_raw_list


def _read_ner_data_to_trainable(path: str, has_label: bool, use_utf16_encoding: bool):
    if use_utf16_encoding:
        with io.open(path, 'r', encoding='utf-16-le') as data_file:
            ner_raw_data = data_file.read()
    else:
        with open(path, 'r') as data_file:
            ner_raw_data = data_file.read()

    raw_sentences = ner_raw_data.split('\n\n')

    dataset_list = []

    for sentence in raw_sentences:
        trainable_sentence = []
        if sentence == '\n' or sentence == '':
            break
        if has_label:
            for word_label in sentence.split('\n'):
                # not sure why \ufeff will be in front of a sentence
                word, label = word_label.strip('\ufeff\n ').split()
                trainable_sentence.append((word, label))
        else:
            for word in sentence.split('\n'):
                word = word.strip('\ufeff\n ')
                trainable_sentence.append(word)

        dataset_list.append(trainable_sentence)

    # List[List[Tuple[str, str]]] or List[List[str]]
    return dataset_list


def get_ner_labels_from_file(path: str, use_utf16_encoding: bool) -> List[List[str]]:
    """ extract all the label for each sentence. lists of per sentence per list """
    dataset_list = _read_ner_data_to_trainable(
        path, has_label=True, use_utf16_encoding=use_utf16_encoding)
    labels = [[label for (_, label) in sentence] for sentence in dataset_list]
    return labels


def read_ner_data_and_split(path: str = RAW_DATA.NER, test_ratio: float = 0.3, use_utf16_encoding: bool = True):
    ner_trainable = _read_ner_data_to_trainable(
        path, has_label=True, use_utf16_encoding=use_utf16_encoding)

    total_lines = len(ner_trainable)
    split_index = round(test_ratio * total_lines)

    np.random.shuffle(ner_trainable)
    train_lines, test_lines = ner_trainable[:-
                                            split_index], ner_trainable[-split_index:]

    return train_lines, test_lines


def from_trainable_to_ner_list(dataset_list: List[List[Tuple[str, str]]], output_path: str = ''):
    ner_list = []
    for sentence in dataset_list:
        sentence_lines = ""
        for (word, label) in sentence:
            sentence_lines += word + ' ' + label + '\n'
        sentence_lines += '\n'

        ner_list.append(sentence_lines)

    if output_path:
        with open(output_path, 'w') as f_out:
            f_out.writelines(ner_list)

    return ner_list


# Deprecated: can't deal with OOV word
def from_ner_numpy_to_evaluable_format(x, y, seq_len, word_to_id: Dict[str, int], tag_to_id: Dict[str, int] = NER_LabelEncode, output_path: str = ''):
    all_sentences = from_numpy_to_trainable(
        x, y, seq_len, word_to_id, tag_to_id)
    from_trainable_to_ner_list(
        all_sentences, output_path)  # just for output file
    return [[label for (_, label) in sentence] for sentence in all_sentences]

def combine_ner_numpy_pred_to_evaluable_format(dataset_list, pred, tag_to_id: Dict[str, int] = NER_LabelEncode, output_path: str = ''):
    all_sentences = combine_numpy_pred_to_trainable(dataset_list, pred, tag_to_id)
    from_trainable_to_ner_list(
        all_sentences, output_path)  # just for output file
    return [[label for (_, label) in sentence] for sentence in all_sentences]

# General usage

def get_total_word_set(train_all_dataset_list: list, fixed_max_seq_len: int = 0):
    """ statistic all the words in the training set and find the max sequence length and the word to id dict """
    all_word_in_train = [
        word for sentence in train_all_dataset_list for (word, _) in sentence]
    if fixed_max_seq_len > 0:
        max_seq_len = fixed_max_seq_len
    else:
        max_seq_len = max([len(sentence)
                           for sentence in train_all_dataset_list])
    # max_seq_len = 165  # this is the max sequence length in the cws test data to predict
    print('Max sentence (sequence) length:', max_seq_len)
    word_set = ['PAD'] + list(set(all_word_in_train))
    print('Total unique word (include PAD):', len(word_set))
    word_to_id = {word: index for index, word in enumerate(word_set)}

    return word_to_id, max_seq_len, word_set


def train_test_trainable_to_numpy(train_dataset_list: list, test_dataset_list: list, train_all_dataset_list: list, tag_to_id: Dict[str, int], fixed_max_seq_len: int = 0):
    """ transfer all the training data into numpy array """
    word_to_id, max_seq_len, _ = get_total_word_set(
        train_all_dataset_list, fixed_max_seq_len=fixed_max_seq_len)

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


def raw_to_numpy(raw_data_line_list: List[str], train_all_dataset_list: list, fixed_max_seq_len: int = 0):
    word_to_id, max_seq_len, _ = get_total_word_set(
        train_all_dataset_list, fixed_max_seq_len=fixed_max_seq_len)

    x = []
    seq_len = []
    for raw_line in raw_data_line_list:
        sentence = [*raw_line]
        x.append([word_to_id[word] if word in word_to_id else word_to_id['PAD'] for word in sentence] +
                 [word_to_id['PAD']] * (max_seq_len - len(sentence)))
        seq_len.append(len(sentence))
        # assert the predict sentence will always less than the max length sentence in training data
        assert len(sentence) <= max_seq_len

    return np.array(x), np.array(seq_len)


# Deprecated: can't deal with OOV word
def from_numpy_to_trainable(x, y, seq_len, word_to_id: Dict[str, int], tag_to_id: Dict[str, int]) -> List[List[Tuple[str, str]]]:
    id_to_tag = {index: tag for tag, index in tag_to_id.items()}
    id_to_word = {index: word for word, index in word_to_id.items()}
    all_sentences = []
    for single_x, single_y, single_len in zip(x, y, seq_len):
        sentence = []
        for i in range(single_len):
            sentence.append((id_to_word[single_x[i]], id_to_tag[single_y[i]]))
        all_sentences.append(sentence)
    return all_sentences

def combine_numpy_pred_to_trainable(dataset_list, pred, tag_to_id: Dict[str, int]) -> List[List[Tuple[str, str]]]:
    id_to_tag = {index: tag for tag, index in tag_to_id.items()}
    all_sentences = []
    for row, ori_sentence in enumerate(dataset_list):
        sentence_pred = []
        if not ori_sentence:
            continue # prevent from empty sentence
        if len(ori_sentence[0]) > 1: # trainable format: (word, label)
            for col, (word, _) in enumerate(ori_sentence):
                sentence_pred.append((word, id_to_tag[pred[row][col]]))
        else: # raw format: word
            for col, word in enumerate(ori_sentence):
                sentence_pred.append((word, id_to_tag[pred[row][col]]))

        all_sentences.append(sentence_pred)

    return all_sentences
            

def CWS_functionality_test():
    print("Get trainable cws data")
    train_data_list, test_data_list, train_all_list, final_raw_list = setup_cws_data()

    print("Test raw data to numpy without label")
    final_x, final_seq_len = raw_to_numpy(
        final_raw_list, train_all_list, fixed_max_seq_len=165)
    print(final_x)
    print(final_seq_len)

    print("Transfer trainable cws data into numpy format")
    (train_x, train_y, train_seq_len), _, _ = train_test_trainable_to_numpy(
        train_data_list, test_data_list, train_all_list, CWS_LabelEncode, fixed_max_seq_len=165)
    print(train_x)

    # word_to_id, _, _ = get_total_word_set(
    #     train_all_list, fixed_max_seq_len=165)  # get word_to_id

    print("Transfer numpy format back to trainable cws format")
    # new_train_data_list = from_numpy_to_trainable(
    #     train_x, train_y, train_seq_len, word_to_id, CWS_LabelEncode)
    new_train_data_list = combine_numpy_pred_to_trainable(
        train_data_list, train_y, CWS_LabelEncode)

    print("Transfer cws format back to raw data")
    cws_list = from_trainable_to_cws_list(
        new_train_data_list, output_path='cws_func_test.txt')

    from evaluation import wordSegmentEvaluaiton
    wordSegmentEvaluaiton(cws_list, cws_list)


def NER_functionality_test():
    print("Get trainable ner data")
    train_data_list, test_data_list, train_all_data_list, final_raw_list = setup_ner_data()

    print("Test raw data to numpy without label")
    final_x, final_seq_len = raw_to_numpy(final_raw_list, train_all_data_list)
    print(final_x)
    print(final_seq_len)

    print("Transfer trainable ner data into numpy format")
    (train_x, train_y, train_seq_len), _, _ = train_test_trainable_to_numpy(
        train_data_list, test_data_list, train_all_data_list, NER_LabelEncode)
    print(train_x)

    # word_to_id, _, _ = get_total_word_set(
    #     train_all_data_list)  # get word_to_id

    print("Transfer numpy format back to trainable ner format")
    # new_train_data_list = from_numpy_to_trainable(
    #     train_x, train_y, train_seq_len, word_to_id, NER_LabelEncode)
    new_train_data_list = combine_numpy_pred_to_trainable(
        train_data_list, train_y, NER_LabelEncode)

    print("Transfer ner format back to raw data")
    ner_list = from_trainable_to_ner_list(
        new_train_data_list, output_path='ner_func_test.txt')


if __name__ == "__main__":
    os.makedirs('train_test', exist_ok=True)

    CWS_functionality_test()
    NER_functionality_test()
