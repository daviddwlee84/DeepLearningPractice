## Chinese Word Segmentation, POS Tagging, Medical NER All In One
# Originally, we have the following procedure
#
# I. Word Segmentation
# II. POS tagging with general NER
# III. Medical NER
#
# But by using tools, I've changed the processing oder
#
# Data clean up ($$_ space)
# I. Word Segmentation and Pre-POS tagging
# II. POS tagging and General NER
# III. Medical NER

# TODO Fix <sub> <sup> tag bug on line 67 of raw_59.txt
# TODO Medical NER: Multiple word prefix, postfix user dictionary

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
        # '安全性', # TA said don't split XX性 = =, but the given example need to split WTF
        '应予以',
        '常继发',
        # '迟发性',
        # '灵敏性',
        '若有阳',
        '完全恢复',
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

def dumpResultIntoTxt(result_dict:dict, data_path:str="2nd_59.txt"):

    print("Dumping result into {}...".format(data_path))
    with open(data_path, 'w') as result:
        for seq_num, string in result_dict.items():
            result.write(str(seq_num) + ' ' + string + '\n')


## Data clean up

def deleteMeaninglessSpace(raw_data_dict:dict):
    print("Deleting meaningless spaces...")
    cleaned_raw_data_dict = {}

    for seq_num, string in raw_data_dict.items():
        # Delete meaningless '$$_'
        # all the $$_ surrounding by digital
        cleaned_text = re.sub(r'(\d.)\$\$_(\d)', r'\1\2', string)

        cleaned_raw_data_dict[seq_num] = cleaned_text
            
    return cleaned_raw_data_dict

def _loadSubSupDict(data_path:str='user_dict/sub_sup.txt'):
    with open(data_path, 'r') as f:
        sub_sup_raw = f.readlines()

    sub_sup_dictionary = {}

    for line in sub_sup_raw:
        sub_sup_word, tag = line.split()
        sub_sup_dictionary[sub_sup_word] = tag

    return sub_sup_dictionary

# TODO The <sub> <sup> tag bug
def preserveSubSupTags(cleaned_raw_data_dict:dict):
    
    print("Preserving <sub> <sup> tags...")
    sub_sup_dictionary = _loadSubSupDict()

    sub_sup_insert_index_dict = defaultdict(list) # seq_num: [(word_position, sub_sup_word, tag), ...]

    no_sub_sup_raw_data_dict = {}

    for seq_num, string in cleaned_raw_data_dict.items():
        for sub_sup_word, tag in sub_sup_dictionary.items():
            sub_sup_pattern = re.escape(sub_sup_word)
            while sub_sup_word in string: # may be same token in same line
                start_idx = string.find(sub_sup_word)
                sub_sup_insert_index_dict[seq_num].append((start_idx, sub_sup_word, tag))
                string = re.sub(sub_sup_pattern, '', string, count=1) # delete the first match
        no_sub_sup_raw_data_dict[seq_num] = string
    
    return no_sub_sup_raw_data_dict, sub_sup_insert_index_dict

# TODO The <sub> <sup> tag bug
def insertSubSupTags(word_list_dict:dict, sub_sup_insert_index_dict:dict, with_tag:bool=True):
    print("Inserting <sub> <sup> with tag..." if with_tag else "Inserting <sub> <sup> without tag...")
    new_word_list_dict = {}
    for seq_num, word_list in word_list_dict.items():
        if seq_num in sub_sup_insert_index_dict:
            cumulated = 0
            new_word_list = word_list.copy()
            for (start_idx, sub_sup_word, tag) in sub_sup_insert_index_dict[seq_num]:
                word_count = 0
                for i, word in enumerate(word_list):
                    if with_tag: # remove tag if the input word_list_dict is after POS tagging
                        if word == '$$_':
                            word = '$$_'
                        else:
                            try:
                                word, _ = word.split('/')
                            except ValueError: # jieba: "//x"
                                word, _ = re.sub(r'(.*)/(\w+)', r'\1 \2', word).split()
                    if word_count == start_idx:
                        if with_tag:
                            new_word_list.insert(i+cumulated, sub_sup_word+'/'+tag)
                        else:
                            new_word_list.insert(i+cumulated, sub_sup_word)
                        break
                    else:
                        word_count += len(word)
                cumulated += 1
            new_word_list_dict[seq_num] = new_word_list
        else:
            new_word_list_dict[seq_num] = word_list

    new_word_dict = {}
    for seq_num, new_word_list in new_word_list_dict.items():
        new_word_dict[seq_num] = " ".join(new_word_list)

    return new_word_dict, new_word_list_dict

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
        
            # only work with jieba (pkuseg will change $$_ to $$&...)
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
        'ag': 'a',
        'df': 'd',
        'dg': 'd',
        'eng': 'nx', # English not nr is nx
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

            if tag == 'x': # TODO
                if word in ('，', '：', '。', '（', '）', '；', '/', '［', '］', '＜', '＞', '、'):  # symbol
                    tag = 'w'
                elif word in ('①', '②', '③', '④', '⑤', '⑥', '⑦'):  # number symbol
                    tag = 'm'
                
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
    
    # Use list to make sure the postfix and prefix pattern will be examined later than normal
    # (NER, (tag, postfix_prefix_normal))
    medical_dictionary_list = []

    for line in medical_raw:
        ner, tag = line.split()
        if ner[0] == '_': # postfix
            medical_dictionary_list.append((ner[1:], (tag, 'postfix')))
        elif ner[-1] == '_': # prefix
            medical_dictionary_list.append((ner[:-1], (tag, 'prefix')))
        else:
            medical_dictionary_list.append((ner, (tag, 'normal')))
    
    return medical_dictionary_list

# Key function for Medical NER 
def _findToMarkPosition(word_seg_list_dict:dict, medical_dictionary_list:list):
    to_mark_list_dict = defaultdict(list) # seq_num: [((pos_start, pos_end), tag)] 

    for seq_num, word_seg_list in word_seg_list_dict.items():
        for i, word in enumerate(word_seg_list):
            for (ner, (tag, post_pre_fix)) in medical_dictionary_list:
                if post_pre_fix == 'normal' and ner == word:
                    to_mark_list_dict[seq_num].append(((i, i), tag))
                elif post_pre_fix == 'normal' and word in ner: # ner is combined with multiple word
                    candidate_word = word
                    for j in range(i+1, len(word_seg_list)):
                        candidate_word += word_seg_list[j]
                        if len(candidate_word) > len(ner):
                            break
                        elif len(candidate_word) == len(ner): # candidate
                            if candidate_word == ner:
                                to_mark_list_dict[seq_num].append(((i, j), tag))
                elif post_pre_fix == 'postfix' and ner in word: # TODO
                    if word[-len(ner):] == ner: # Current only support single word postfix
                        if ((i, i), tag) not in to_mark_list_dict[seq_num]: # make sure not duplicate (first filtering)
                            found_duplicate_entity = False
                            for ((ii, jj), _) in to_mark_list_dict[seq_num]:
                                # If there is a same word but with same postfix (ii)
                                # e.g. _水肿 vs. 肺水肿
                                # If there is a longer word then they will have the same ending index (jj)
                                # e.g. 细菌性心内膜炎 vs. _炎 (will get 心内膜炎)
                                if i == ii or i == jj:
                                    found_duplicate_entity = True
                            if not found_duplicate_entity:
                                to_mark_list_dict[seq_num].append(((i, i), tag))
                elif post_pre_fix == 'prefix' and ner in word:
                    if word[:len(ner)] == ner:
                        if ((i, i), tag) not in to_mark_list_dict[seq_num]:
                            to_mark_list_dict[seq_num].append(((i, i), tag))

    return to_mark_list_dict

def medicalNER(inserted_word_seg_list_dict:dict, inserted_new_pos_list_dict:dict, tools:str='jieba'):
    assert tools in ('pkuseg', 'jieba') # no difference between two tools for now
    print("Medical NER using {}...".format(tools))

    medical_dictionary_list = _loadMedicalDict()

    to_mark_list_dict = _findToMarkPosition(inserted_word_seg_list_dict, medical_dictionary_list)

    medical_ner_list_dict = {}

    for seq_num in inserted_new_pos_list_dict.keys():
        medical_list = inserted_new_pos_list_dict[seq_num]

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
# Gold: 計算機 總是 有問題  => (1, 4) (4, 6) (6, 9)
# Predict: 計算機 總 是 有問題 => (1, 4) (4, 5) (5, 6) (6, 9)
# correct: (1, 4) (6, 9) error: (4, 5) (5, 6)
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
    no_sub_sup_raw_data_dict, sub_sup_insert_index_dict = preserveSubSupTags(cleaned_raw_data_dict)

    ## Prediction
    # I. Word Segmentation and Pre-POS
    word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(no_sub_sup_raw_data_dict, tools='jieba')
    inserted_word_seg_dict, inserted_word_seg_list_dict = insertSubSupTags(word_seg_list_dict, sub_sup_insert_index_dict, with_tag=False)
    dumpResultIntoTxt(inserted_word_seg_dict, data_path='2nd_59_segment.txt')

    # II. POS tagging and General NER

    new_pos_dict, new_pos_list_dict = posWithGeneralNER(pre_pos_list_dict, tools='jieba')
    inserted_new_pos_dict, inserted_new_pos_list_dict = insertSubSupTags(new_pos_list_dict, sub_sup_insert_index_dict, with_tag=True)
    dumpResultIntoTxt(inserted_new_pos_dict, data_path='2nd_59_pos.txt')

    ## III. Medical NER
    medical_ner_dict, medical_ner_list_dict = medicalNER(inserted_word_seg_list_dict, inserted_new_pos_list_dict)

    ## Export Result
    dumpResultIntoTxt(medical_ner_dict)

    ## Evaluation using pdf example
    raw_data_path = 'sample_data/raw.txt'

    seg_ans_path = 'sample_data/segment.txt'
    pos_ans_path = 'sample_data/pos.txt'
    ner_ans_path = 'sample_data/ner.txt'

    raw_data_dict = loadRawDataIntoDict(raw_data_path)
    cleaned_raw_data_dict = deleteMeaninglessSpace(raw_data_dict)
    # preserveSubSupTags()
    _, jieba_word_seg_list_dict, _, jieba_pre_pos_list_dict = wordSegmentationWithPOS(
        cleaned_raw_data_dict, tools='jieba')
    # insertSubSupTags()
    _, pkuseg_word_seg_list_dict, _, pkuseg_pre_pos_list_dict = wordSegmentationWithPOS(
        cleaned_raw_data_dict, tools='pkuseg')
    # insertSubSupTags()
    _, jieba_pos_list_dict = posWithGeneralNER(jieba_pre_pos_list_dict, tools='jieba')
    # insertSubSupTags()
    _, pkuseg_pos_list_dict = posWithGeneralNER(pkuseg_pre_pos_list_dict, tools='pkuseg')
    # change input to inserted word_seg and pos list_dict
    _, jieba_medical_ner_list_dict = medicalNER(jieba_word_seg_list_dict, jieba_pos_list_dict)
    # change input to inserted word_seg and pos list_dict
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
