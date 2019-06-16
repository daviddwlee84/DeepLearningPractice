from dataset import load_utf16le_data_to_list
from constant import RAW_DATA
from typing import List

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


if __name__ == "__main__":
    # test evaluation function
    test_list = load_utf16le_data_to_list(RAW_DATA.CWS)
    wordSegmentEvaluaiton(test_list, test_list)
