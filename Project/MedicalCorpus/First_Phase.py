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


import jieba
from collections import defaultdict
import re # Regular Expression

# Data Processing
from pkuseg import pkuseg
pseg = pkuseg(model_name='medicine', postag=True,
              user_dict='user_dict/user_dict.txt')
jieba.load_userdict('user_dict/user_dict.txt')
import jieba.posseg as jseg

def _jiebaPOSRule():
    needAdd = [
        ('10<sup>12</sup>', 'w'), # symbol didn't work (TOTO)
        ('Ca<sup>2+</sup>', 'n'),
        ('10<sup>9</sup>', 'w'),
        ('<sup>*</sup>', 'x'),
        ('PaO<sub>2</sub>', 'n'),
        ('CO<sub>2</sub>', 'n'),
        ('U<sub>1</sub>', 'n'),
        ('PaO<sub>2</sub>', 'n'),
        ('PaCO<sub>2</sub>', 'n'),
        ('PaO<sub>2</sub>', 'n'),
        ('PaCO<sub>2</sub>', 'n'),
        ('CD<sub>33</sub>', 'n'),
        ('CD<sub>13</sub>', 'n'),
        ('CD<sub>15</sub>', 'n'),
        ('CD<sub>11</sub>b', 'n'),
        ('CD<sub>36</sub>', 'n'),
    ]
    for add_word, tag in needAdd:
        jieba.add_word(add_word, freq=100, tag=tag)

    needRetain = [
        '去大脑',
        '广谱', # 广谱抗生素
        '阳转',
    ]
    for retain_word in needRetain:
        jieba.suggest_freq(retain_word, tune=True)

    needExtract = [
        '体格检查',
        '光反应',
        '对光',
        '创伤性',
        '细菌性',
        '行为矫正',
        '粟粒状',
        '安全性',
        '应予以',
        '常继发',
        '迟发性',
        '灵敏性',
        '若有阳',
        'sup', # english didn't work (TOTO)
        'sub',
    ]
    for del_word in needExtract:
        jieba.del_word(del_word)

lastName = ['赵', '钱', '孙', '李', '周', '吴', '郑', '王', '冯', '陈', '褚', '卫', '蒋', '沈', '韩', '杨', '朱', '秦', '尤', '许',
            '何', '吕', '施', '张', '孔', '曹', '严', '华', '金', '魏', '陶', '姜', '戚', '谢', '邹', '喻', '柏', '水', '窦', '章',
            '云', '苏', '潘', '葛', '奚', '范', '彭', '郎', '鲁', '韦', '昌', '马', '苗', '凤', '花', '方', '俞', '任', '袁', '柳',
            '酆', '鲍', '史', '唐', '费', '廉', '岑', '薛', '雷', '贺', '倪', '汤', '滕', '殷', '罗', '毕', '郝', '邬', '安', '常',
            '乐', '于', '时', '傅', '皮', '卞', '齐', '康', '伍', '余', '元', '卜', '顾', '孟', '平', '黄', '和', '穆', '萧', '尹',
            '姚', '邵', '堪', '汪', '祁', '毛', '禹', '狄', '米', '贝', '明', '臧', '计', '伏', '成', '戴', '谈', '宋', '茅', '庞',
            '熊', '纪', '舒', '屈', '项', '祝', '董', '梁', '司马']

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
            result.write(str(seq_num) + ' ' + string + '\n')


## Data clean up

def deleteMeaninglessSpace(raw_data_dict:dict):
    cleaned_raw_data_dict = {}

    for seq_num, string in raw_data_dict.items():
        # Delete meaningless '$$_'
        # all the $$_ surrounding by digital
        cleaned_text = re.sub(r'(\d.)\$\$_(\d)', r'\1\2', string)

        cleaned_raw_data_dict[seq_num] = cleaned_text
            
    return cleaned_raw_data_dict

## I. Word Segmentation and Pre-POS tagging

def _firstWordSegmentationWithPOS(cleaned_raw_data_dict:dict, tools:str='jieba'):
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

        spaceDetector = 0
        for word, flag in words:
            word_with_tag = word + '/' + flag

            if word == '\n': # jieba will retain last \n as word
                continue
            
            if flag == 'nr': # people name
                # print(word, flag) # useful to find mis-classified name
                if len(word) >= 2 and word[0:2] in lastName:  # e.g. 司馬
                    word_seg_list_dict[seq_num].append(word[0:2])
                    pre_pos_list_dict[seq_num].append(word[0:2]+'/nr')
                    word = word[2:]
                    word_with_tag = word_with_tag[2:]
                    if len(word) == 2: # only lastname
                        continue
                elif word[0] in lastName:
                    word_seg_list_dict[seq_num].append(word[0])
                    pre_pos_list_dict[seq_num].append(word[0]+'/nr')
                    if len(word) == 1: # only lastname
                        continue
                    word = word[1:]
                    word_with_tag = word_with_tag[1:]

            word_seg_list_dict[seq_num].append(word)
            pre_pos_list_dict[seq_num].append(word_with_tag)
        
            # only work with jieba
            if word == '$' and spaceDetector == 0:
                spaceDetector += 1
            elif word == '$' and spaceDetector == 1:
                spaceDetector += 1
            elif word == '_' and spaceDetector == 2:
                spaceDetector = 0
                for _ in range(3):
                    word_seg_list_dict[seq_num].pop()
                    pre_pos_list_dict[seq_num].pop()
                word_seg_list_dict[seq_num].append('$$_')
                pre_pos_list_dict[seq_num].append('$$_')
                spaceDetector = 0
            else:
                spaceDetector = 0

    for seq_num in word_seg_list_dict.keys():
        word_seg_dict[seq_num] = " ".join(word_seg_list_dict[seq_num])
        pre_pos_dict[seq_num] = " ".join(pre_pos_list_dict[seq_num])

    return word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict

def wordSegmentationWithPOS(cleaned_raw_data_dict:dict, tools:str='jieba'):
    word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = _firstWordSegmentationWithPOS(cleaned_raw_data_dict, tools)
    return word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict


## II. POS tagging and General NER

def _jiebaPOSMapper(pre_pos_list_dict:dict):

    jiebaExtraTags = {
        'ad': 'a',
        'an': 'a',
        'df': 'd',
        'dg': 'd',
        'mg': 'm',
        'mq': 'm',
        'ng': 'n',
        'nrfg': 'n',
        'nrt': 'n',
        'nz': 'n',
        'rg': 'r',
        'rr': 'r',
        'rz': 'r',
        'tg': 't',
        'ud': 'u',
        'ug': 'u',
        'uj': 'u',
        'ul': 'u',
        'uv': 'u',
        'uz': 'u',
        'vd': 'v',
        'vg': 'v',
        'vi': 'v',
        'vn': 'v',
        'vq': 'v',
        'zg': 'z'
    }

    new_pos_list_dict = {}
    for seq_num, pre_pos_list in pre_pos_list_dict.items():
        new_pos_list = []
        for word_with_tag in pre_pos_list:
            if word_with_tag == '$$_':
                new_pos_list.append('$$_')
                continue
            try:
                word, tag = word_with_tag.split('/')
            except ValueError: # jieba: "//x"
                word, tag = re.sub(r'(.*)/(\w+)', r'\1 \2', word_with_tag).split()
            if tag in jiebaExtraTags:
                tag = jiebaExtraTags[tag] # equivalent to pick first char
            new_word_with_tag = word + '/' + tag
            new_pos_list.append(new_word_with_tag)
        new_pos_list_dict[seq_num] = new_pos_list

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
            if word_with_tag == '$$_':
                new_pos_list.append('$$_')
                continue
            word, tag = word_with_tag.split('/')
            if tag in pkusegExtraTags:
                tag = pkusegExtraTags[tag]
            new_word_with_tag = word + '/' + tag
            new_pos_list.append(new_word_with_tag)
        new_pos_list_dict[seq_num] = new_pos_list

    return new_pos_list_dict

def posWithGeneralNER(pre_pos_list_dict:dict, tools:str='jieba'):
    assert tools in ('pkuseg', 'jieba')
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

def _loadMedicalDict(data_path:str='user_dict/medical_ner.txt'):
    with open(data_path, 'r') as f:
        medical_raw = f.readlines()
    
    # (NER: (tag, postfix_prefix_normal))
    medical_dictionary = {}

    for line in medical_raw:
        ner, tag = line.split()
        if ner[0] == '_': # postfix
            medical_dictionary[ner[1:]] = (tag, 'postfix')
        elif ner[-1] == '_': # prefix
            medical_dictionary[ner[:-1]] = (tag, 'prefix')
        else:
            medical_dictionary[ner] = (tag, 'normal')
    
    return medical_dictionary
            
def _findToMarkPosition(word_seg_list_dict:dict, medical_dictionary: dict):
    to_mark_list_dict = defaultdict(list) # seq_num: [((pos_start, pos_end), tag)] 

    for seq_num, word_seg_list in word_seg_list_dict.items():
        for i, word in enumerate(word_seg_list):
            for ner, (tag, post_pre_fix) in medical_dictionary.items():
                if post_pre_fix == 'normal' and ner == word:
                    to_mark_list_dict[seq_num].append(((i, i), tag))
                elif post_pre_fix == 'postfix' and ner in word:
                    pass
                elif post_pre_fix == 'prefix' and ner in word:
                    pass

    return to_mark_list_dict

def medicalNER(word_seg_list_dict: dict, new_pos_list_dict:dict, tools:str='jieba'):
    assert tools in ('pkuseg', 'jieba') # no difference between two tools for now
    print("Medical NER using {}...".format(tools))

    medical_dictionary = _loadMedicalDict()

    to_mark_list_dict = _findToMarkPosition(word_seg_list_dict, medical_dictionary)

    medical_ner_list_dict = {}

    for seq_num in new_pos_list_dict.keys():
        medical_list = new_pos_list_dict[seq_num]

        if seq_num in to_mark_list_dict:
            for (pos_start, pos_end), tag in to_mark_list_dict[seq_num]:
                medical_list[pos_start] = '[' + medical_list[pos_start]
                medical_list[pos_end] = medical_list[pos_end] + ']' + tag

        medical_ner_list_dict[seq_num] = medical_list
            
    medical_ner_dict = {}
    for seq_num, medical_ner_list in medical_ner_list_dict.items():
        medical_ner_dict[seq_num] = " ".join(medical_ner_list)

    return medical_ner_dict, medical_ner_list_dict


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

def _scorerSingle(pred_eval_list:list, gold_eval_list:list, print_fail_num:int=0):
    e = 0
    c = 0
    N = len(gold_eval_list)

    for pred_start_end in pred_eval_list:
        if pred_start_end in gold_eval_list:
            c += 1
        else:
            e += 1
            if print_fail_num:
                print('line:', print_fail_num, 'found error:',
                      pred_start_end, '=>', debugHelper(print_fail_num, *pred_start_end, 'sample_data/raw.txt'))
                
    
    return e, c, N

def _scorer(pred_eval_list_dict:dict, gold_eval_list_dict:dict, print_fail:bool=False):
    N = 0 # gold segment words number
    e = 0 # wrong number of word segment
    c = 0 # correct number of word segment

    for seq_num in gold_eval_list_dict.keys():
        pred_eval_list = pred_eval_list_dict[seq_num]
        gold_eval_list = gold_eval_list_dict[seq_num]
        if print_fail:
            temp_e, temp_c, temp_N = _scorerSingle(pred_eval_list, gold_eval_list, int(seq_num))
        else:
            temp_e, temp_c, temp_N = _scorerSingle(pred_eval_list, gold_eval_list)

        N += temp_N
        e += temp_e
        c += temp_c
    
    R = c/N
    P = c/(c+e)
    F1 = (2*P*R)/(P+R)
    ER = e/N

    return R, P, F1, ER

def wordSegmentEvaluaiton(pred_list_dict:dict, gold_list_dict:dict, print_fail:bool=False):

    pred_eval_list_dict = {}
    gold_eval_list_dict = {}
    for seq_num in gold_list_dict.keys():
        pred_list = pred_list_dict[seq_num]
        gold_list = gold_list_dict[seq_num]
        
        pred_eval_list_dict[seq_num] = _toSegEvalFormat(pred_list)
        gold_eval_list_dict[seq_num] = _toSegEvalFormat(gold_list)
    
    P, R, F1, ER = _scorer(pred_eval_list_dict, gold_eval_list_dict, print_fail)
        
    print('=== Evaluation reault of word segment ===')
    print('F1: %.2f%%' % (F1*100))
    print('P : %.2f%%' %  (P*100))
    print('R : %.2f%%' %  (R*100))
    print('ER: %.2f%%' %  (ER*100))
    print('=========================================')


# Print the word index that _scorer said. (if you enable print_fail of wordSegmentEvaluaiton)
def debugHelper(seq_num:int, start_num:int, end_num:int, raw_data_path:str):
    # PS. start from 1
    with open(raw_data_path, 'r') as f:
        lines = f.readlines()
    _, string = lines[seq_num-1].split(' ', 1) # skip the line number
    # print('line:', seq_num, (start_num, end_num), '=>', string[start_num-1:end_num-1])
    return string[start_num-1:end_num-1]

def main():

    ## Loading data
    raw_data_path = 'data/raw_59.txt'
    raw_data_dict = loadRawDataIntoDict(raw_data_path)

    # delete $$_ before I.
    cleaned_raw_data_dict = deleteMeaninglessSpace(raw_data_dict)

    ## Prediction
    # I. Word Segmentation and Pre-POS
    word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(cleaned_raw_data_dict, tools='jieba')
    # word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(cleaned_raw_data_dict, tools='pkuseg')
    dumpResultIntoTxt(word_seg_dict, data_path='1_ 59_segment.txt')

    # II. POS tagging and General NER

    new_pos_dict, new_pos_list_dict = posWithGeneralNER(pre_pos_list_dict, tools='jieba')
    # new_pos_dict, new_pos_list_dict = posWithGeneralNER(pre_pos_list_dict, tools='pkuseg')
    dumpResultIntoTxt(new_pos_dict, data_path='1_ 59_pos.txt')

    ## III. Medical NER
    medical_ner_dict, medical_ner_list_dict = medicalNER(word_seg_list_dict, new_pos_list_dict)

    ## Export Result
    dumpResultIntoTxt(medical_ner_dict)

    ## Evaluation using pdf example
    raw_data_path = 'sample_data/raw.txt'

    seg_ans_path = 'sample_data/segment.txt'
    pos_ans_path = 'sample_data/pos.txt'
    ner_ans_path = 'sample_data/ner.txt'

    raw_data_dict = loadRawDataIntoDict(raw_data_path)
    cleaned_raw_data_dict = deleteMeaninglessSpace(raw_data_dict)
    _, jieba_word_seg_list_dict, _, jieba_pre_pos_list_dict = wordSegmentationWithPOS(
        cleaned_raw_data_dict, tools='jieba')
    _, pkuseg_word_seg_list_dict, _, pkuseg_pre_pos_list_dict = wordSegmentationWithPOS(
        cleaned_raw_data_dict, tools='pkuseg')
    _, jieba_pos_list_dict = posWithGeneralNER(jieba_pre_pos_list_dict, tools='jieba')
    _, pkuseg_pos_list_dict = posWithGeneralNER(pkuseg_pre_pos_list_dict, tools='pkuseg')
    _, jieba_medical_ner_list_dict = medicalNER(jieba_word_seg_list_dict, jieba_pos_list_dict)
    _, pkuseg_medical_ner_list_dict = medicalNER(pkuseg_word_seg_list_dict, pkuseg_pos_list_dict)
    seg_ans_list_dict, pos_ans_list_dict, ner_ans_list_dic = loadAnswerIntoDict(seg_ans_path, pos_ans_path, ner_ans_path)

    print('Test jieba word segmentation')
    wordSegmentEvaluaiton(jieba_word_seg_list_dict, seg_ans_list_dict, print_fail=True)
    print('Test pkuseg word segmentation')
    wordSegmentEvaluaiton(pkuseg_word_seg_list_dict, seg_ans_list_dict, print_fail=True)

    print('\nGold segment:\n', seg_ans_list_dict)
    print('\njieba:\n', jieba_word_seg_list_dict)
    print('\npkuseg:\n', pkuseg_word_seg_list_dict)

    print('\nGold POS:\n', pos_ans_list_dict)
    print('\njieba:\n', jieba_pos_list_dict)
    print('\npkuseg:\n', pkuseg_pos_list_dict)

    print('\nGold medical NER:\n', ner_ans_list_dic)
    print('\njieba:\n', jieba_medical_ner_list_dict)
    print('\npkuseg:\n', pkuseg_medical_ner_list_dict)

if __name__ == "__main__":
    _jiebaPOSRule()
    main()
