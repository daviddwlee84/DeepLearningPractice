from dataset import load_utf16le_data_to_list, get_ner_labels_from_file, NER_LabelEncode
from constant import RAW_DATA
from typing import List
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

# CWS Function

# from segment to evaluation-able format
# e.g.
# Gold: 計算機 總是 有問題  => (1, 4) (4, 6) (6, 9)
# Predict: 計算機 總 是 有問題 => (1, 4) (4, 5) (5, 6) (6, 9)
# correct: (1, 4) (6, 9) error: (4, 5) (5, 6)


def _toSegEvalFormat(string_list: List[str]):
    eval_format_list = []
    word_count = 1  # start from 1
    for string in string_list:
        start = word_count
        word_count += len(string)
        end = word_count
        eval_format_list.append((start, end))
    return eval_format_list


def _scorerSingle(pred_eval: List[str], gold_eval: List[str]):
    e = 0
    c = 0
    N = len(gold_eval)

    for pred_start_end in pred_eval:
        if pred_start_end in gold_eval:
            c += 1
        else:
            e += 1

    return e, c, N


def _scorer(pred_eval_list: List[List[str]], gold_eval_list: List[List[str]]):
    N = 0  # gold segment words number
    e = 0  # wrong number of word segment
    c = 0  # correct number of word segment

    for pred_eval, gold_eval in zip(pred_eval_list, gold_eval_list):
        temp_e, temp_c, temp_N = _scorerSingle(pred_eval, gold_eval)

        N += temp_N
        e += temp_e
        c += temp_c

    R = c/N
    P = c/(c+e)
    F1 = (2*P*R)/(P+R)
    ER = e/N

    return R, P, F1, ER


def wordSegmentEvaluaiton(pred_seg_list: List[str], gold_seg_list: List[str]):

    pred_eval_list = []
    gold_eval_list = []
    for pred_string, gold_string in zip(pred_seg_list, gold_seg_list):
        pred_list = pred_string.split()
        gold_list = gold_string.split()
        pred_eval_list.append(_toSegEvalFormat(pred_list))
        gold_eval_list.append(_toSegEvalFormat(gold_list))

    P, R, F1, ER = _scorer(pred_eval_list, gold_eval_list)

    print('=== Evaluation reault of word segment ===')
    print('F1: %.2f%%' % (F1*100))
    print('P : %.2f%%' % (P*100))
    print('R : %.2f%%' % (R*100))
    print('ER: %.2f%%' % (ER*100))
    print('=========================================')

# NER Function


def per_token_eval(pred_ner_labels: List[str], gold_ner_labels: List[str], labels: List[str]):
    labels.remove('N')
    sorted_labels = sorted(labels, key=lambda name: (
        name[1:], name[0]))  # group B and I results
    return flat_classification_report(pred_ner_labels, gold_ner_labels, labels=sorted_labels, digits=4)


def namedEntityEvaluation(pred_ner_labels: List[str], gold_ner_labels: List[str]):
    classes = list(NER_LabelEncode.keys())
    print("Performance per label type per token")
    print(per_token_eval(pred_ner_labels, gold_ner_labels, classes))

    print("Performance over full named-entity")
    # TODO


if __name__ == "__main__":
    print("test cws evaluation function")
    test_list = load_utf16le_data_to_list(RAW_DATA.CWS)
    wordSegmentEvaluaiton(test_list, test_list)

    print("test ner evaluation function")
    test_labels = get_ner_labels_from_file(
        RAW_DATA.NER, use_utf16_encoding=True)
    namedEntityEvaluation(test_labels, test_labels)
