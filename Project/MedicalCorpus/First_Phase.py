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



def main():

    ## Loading data
    raw_data_path = 'data/raw_59.txt'
    # raw_data_path = 'sample_data/raw.txt'

    raw_data_dict = loadRawDataIntoDict(raw_data_path)

    # delete $$_ before I.
    cleaned_raw_data_dict = cleanSpaceUp(raw_data_dict)

    ## Prediction
    # I. Word Segmentation and Pre-POS
    # word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(cleaned_raw_data_dict, tools='jieba')

    # print(word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict)

    word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict = wordSegmentationWithPOS(cleaned_raw_data_dict, tools='pkuseg')

    # print(word_seg_dict, word_seg_list_dict, pre_pos_dict, pre_pos_list_dict)

    # II. POS tagging and General NER

    new_pos_dict, new_pos_list_dict = posWithGeneralNER(pre_pos_list_dict)

    # print(new_pos_dict, new_pos_list_dict)

    ## Export Result
    dumpResultIntoTxt(new_pos_dict)

    ## Evaluation
    seg_ans_path = 'sample_data/segment.txt'
    pos_ans_path = 'sample_data/pos.txt'
    ner_ans_path = 'sample_data/ner.txt'
    seg_ans_list_dict, pos_ans_list_dict, ner_ans_list_dic = loadAnswerIntoDict(seg_ans_path, pos_ans_path, ner_ans_path)

    # print(seg_ans_list_dict, pos_ans_list_dict, ner_ans_list_dic)

if __name__ == "__main__":
    main()
