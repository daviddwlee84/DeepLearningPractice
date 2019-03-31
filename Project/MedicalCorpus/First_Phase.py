## First Phase: using tools and libraries
# Originally, we have the following procedure
#
# I. Word Segmentation
# II. POS tagging with general NER
# III. Medical NER
#
# But by using tools, I've changed the processing oder
#
# (TODO) Data clean up ($$_ space)
# I. Word Segmentation and Pre-POS tagging
# II. POS tagging and General NER
# III. Medical NER


from collections import defaultdict
import re # Regular Expression

# Data Processing
from pkuseg import pkuseg
pseg = pkuseg(model_name='medicine', postag=True)
import jieba.posseg as jseg


## Loading data

def _loadDataIntoListDict(data_path:str):
    with open(data_path, 'r') as data:
        lines = data.readlines()
    
    data_dict = {}
    data_list_dict = {}

    for line in lines:
        # split the sequence number from the context
        seq_num, string = line.split(None, 1)
        data_dict[seq_num] = string
        data_list_dict[seq_num] = string.split()
    
    return data_dict, data_list_dict

def loadRawDataIntoDict(raw_data_path:str):

    print("Loading raw data into dict...")
    raw_data_dict, _ = _loadDataIntoListDict(raw_data_path)

    return raw_data_dict

def loadAnswerIntoDict(seg_ans_path:str, pos_ans_path:str, ner_ans_path:str):

    print("Loading word segmentation answer into dict...")
    _, seg_ans_list_dict = _loadDataIntoListDict(seg_ans_path)

    print("Loading part-of-speech tagging answer into dict...")
    _, pos_ans_list_dict = _loadDataIntoListDict(pos_ans_path)

    print("Loading named-entity recognition answer into dict...")
    _, ner_ans_list_dict = _loadDataIntoListDict(ner_ans_path)

    return seg_ans_list_dict, pos_ans_list_dict, ner_ans_list_dict


## Dumping Data

def dumpResultIntoTxt(result_dict:dict, data_path:str="1_ 59.txt"):

    print("Dumping result into {}...".format(data_path))
    with open(data_path, 'w') as result:
        for seq_num, string in result_dict.items():
            if string[-1] == '\n': # jieba
                result.write(str(seq_num) + ' ' + string)
            else: # pkuseg
                result.write(str(seq_num) + ' ' + string + '\n')


## Data clean up

def cleanSpaceUp(raw_data_dict:dict):
    cleaned_raw_data_dict = {}

    for seq_num, string in raw_data_dict.items():
        # Replace all the '$$_' with ' '
        # all the $$_ not surrounding by digital
        replaced_space = re.sub(r'(\D\D)\$\$_(\D\D)', r'\1 \2', string)
        # all the $$_ surrounding by english letter
        replaced_space = re.sub(r'(\w)\$\$_(\w)', r'\1 \2', replaced_space)

        # Delete all the other '$$_'
        cleaned_text = replaced_space.replace('$$_', '')

        cleaned_raw_data_dict[seq_num] = cleaned_text
            
    return cleaned_raw_data_dict

## I. Word Segmentation and Pre-POS tagging

def _firstWordSegmentationWithPOS(cleaned_raw_data_dict:dict, tools:str='pkuseg'):
    assert tools in ('pkuseg', 'jieba')
    print("Chinese word segmenting and Pre-part-of-speech tagging using {}...".format(tools))

    word_seg_list_dict = defaultdict(list)
    word_seg_dict = {}

    pre_pos_list_dict = defaultdict(list)
    pre_pos_dict = {}

    for seq_num, string in cleaned_raw_data_dict.items():
        
        if tools == 'pkuseg':
            words = pseg.cut(string)
        elif tools == 'jieba':
            words = jseg.cut(string)

        for word, flag in words:
            word_with_tag = word + '/' + flag

            word_seg_list_dict[seq_num].append(word)
            pre_pos_list_dict[seq_num].append(word_with_tag)

        word_seg_dict[seq_num] = " ".join(word_seg_list_dict[seq_num])
        pre_pos_dict[seq_num] = " ".join(pre_pos_list_dict[seq_num])

    return word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict

def wordSegmentationWithPOS(cleaned_raw_data_dict:dict, tools:str='pkuseg'):
    word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = _firstWordSegmentationWithPOS(cleaned_raw_data_dict, tools)
    return word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict


## II. POS tagging and General NER

def _jiebaPOSMapper(pre_pos_list_dict:dict):
    new_pos_list_dict = {}
    return new_pos_list_dict

def _pkusegPOSMapper(pre_pos_list_dict:dict):

    pkusegExtraTags = {
        'nx': 'n',
        'nz': 'n',
        'vd': 'v',
        'vn': 'v',
        'vx': 'v',
        'ad': 'a',
        'an': 'a'
    }

    new_pos_list_dict = {}
    for seq_num, pre_pos_list in pre_pos_list_dict.items():
        new_pos_list = []
        for word_with_tag in pre_pos_list:
            word, tag = word_with_tag.split('/')
            if tag in pkusegExtraTags:
                tag = pkusegExtraTags[tag]
            new_word_with_tag = word + '/' + tag
            new_pos_list.append(new_word_with_tag)
        new_pos_list_dict[seq_num] = new_pos_list

    return new_pos_list_dict

def posWithGeneralNER(pre_pos_list_dict:dict, tools='pkuseg'):
    assert tools in ('pkuseg') # currently only support pkuseg
    print("Part-of-speech tagging with General NER using {}...".format(tools))

    if tools == 'pkuseg':
        new_pos_list_dict = _pkusegPOSMapper(pre_pos_list_dict)
    elif tools == 'jieba':
        new_pos_list_dict = _jiebaPOSMapper(pre_pos_list_dict)

    new_pos_dict = {}
    for seq_num, new_pos_list in new_pos_list_dict.items():
        new_pos_dict[seq_num] = " ".join(new_pos_list)

    return new_pos_dict, new_pos_list_dict

## III. Medical NER


## Evaluation

# from segment to evaluation-able format
# e.g.
# 計算機 總是 有問題  => (1, 4) (4, 6) (6, 9)
# 計算機 總 是 有問題 => (1, 4) (4, 5) (5, 6) (6, 9)
def _toSegEvalFormat(string_list:list):
    eval_format_list = []
    word_count = 1 # start from 1
    for string in string_list:
        start = word_count
        word_count += len(string)
        end = word_count
        eval_format_list.append((start, end))
    return eval_format_list

def _scorerSingle(pred_eval_list:list, gold_eval_list:list):
    e = 0
    c = 0
    N = len(gold_eval_list)

    for pred_start_end in pred_eval_list:
        if pred_start_end in gold_eval_list:
            c += 1
        else:
            e += 1
    
    return e, c, N

def _scorer(pred_eval_list_dict:dict, gold_eval_list_dict:dict):
    N = 0 # gold segment words number
    e = 0 # wrong number of word segment
    c = 0 # correct number of word segment

    for seq_num in gold_eval_list_dict.keys():
        pred_eval_list = pred_eval_list_dict[seq_num]
        gold_eval_list = gold_eval_list_dict[seq_num]
        temp_e, temp_c, temp_N = _scorerSingle(pred_eval_list, gold_eval_list)

        N += temp_N
        e += temp_e
        c += temp_c
    
    R = c/N
    P = c/(c+e)
    F1 = (2*P*R)/(P+R)
    ER = e/N

    return R, P, F1, ER

def wordSegmentEvaluaiton(pred_list_dict:dict, gold_list_dict:dict):

    pred_eval_list_dict = {}
    gold_eval_list_dict = {}
    for seq_num in gold_list_dict.keys():
        pred_list = pred_list_dict[seq_num]
        gold_list = gold_list_dict[seq_num]
        
        pred_eval_list_dict[seq_num] = _toSegEvalFormat(pred_list)
        gold_eval_list_dict[seq_num] = _toSegEvalFormat(gold_list)
    
    P, R, F1, ER = _scorer(pred_eval_list_dict, gold_eval_list_dict)
        
    print('=== Evaluation reault of word segment ===')
    print('F1: %.2f%%' % (F1*100))
    print('P : %.2f%%' %  (P*100))
    print('R : %.2f%%' %  (R*100))
    print('ER: %.2f%%' %  (ER*100))
    print('=========================================')


def main():

    ## Loading data
    raw_data_path = 'data/raw_59.txt'
    raw_data_dict = loadRawDataIntoDict(raw_data_path)

    # delete $$_ before I.
    cleaned_raw_data_dict = cleanSpaceUp(raw_data_dict)

    ## Prediction
    # I. Word Segmentation and Pre-POS
    # word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(cleaned_raw_data_dict, tools='jieba')
    word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(cleaned_raw_data_dict, tools='pkuseg')
    dumpResultIntoTxt(word_seg_dict, data_path='1_ 59_segment.txt')

    # II. POS tagging and General NER

    new_pos_dict, new_pos_list_dict = posWithGeneralNER(pre_pos_list_dict)

    # print(new_pos_dict, new_pos_list_dict)

    ## Export Result
    dumpResultIntoTxt(new_pos_dict)

    ## Evaluation using pdf example
    raw_data_path = 'sample_data/raw.txt'

    seg_ans_path = 'sample_data/segment.txt'
    pos_ans_path = 'sample_data/pos.txt'
    ner_ans_path = 'sample_data/ner.txt'

    raw_data_dict = loadRawDataIntoDict(raw_data_path)
    # cleaned_raw_data_dict = cleanSpaceUp(raw_data_dict)
    _, pkuseg_word_seg_list_dict, _, _ = wordSegmentationWithPOS(raw_data_dict, tools='pkuseg')
    _, jieba_word_seg_list_dict, _, _ = wordSegmentationWithPOS(raw_data_dict, tools='jieba')
    seg_ans_list_dict, pos_ans_list_dict, ner_ans_list_dic = loadAnswerIntoDict(seg_ans_path, pos_ans_path, ner_ans_path)

    print('Test pkuseg word segmentation')
    wordSegmentEvaluaiton(pkuseg_word_seg_list_dict, seg_ans_list_dict)
    print('Test jieba word segmentation')
    wordSegmentEvaluaiton(jieba_word_seg_list_dict, seg_ans_list_dict)

    print('\nGold segment:\n', seg_ans_list_dict)
    print('\npkuseg:\n', pkuseg_word_seg_list_dict)
    print('\njieba:\n', jieba_word_seg_list_dict)

if __name__ == "__main__":
    main()
