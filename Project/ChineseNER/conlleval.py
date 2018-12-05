import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

from collections import defaultdict, namedtuple

Metrics = namedtuple('Metrics', 'prec rec fscore')
label_dict = {}
predict_dict = {}

class Unit:
    def __init__(self):
        self.pre_tag = ""
        self.words = ""
        self.tags = ""

    def printInfo(self):
        print('pre_tag:{}\t words:{}\t tags={}'.format(self.pre_tag, self.words, self.tags))

def calculate_metrics(tp, total_predict, total_label):
    p = 0 if total_predict == 0 else 1.*tp / total_predict
    r = 0 if total_label == 0 else 1.*tp / total_label
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(p, r, f)

def end_of_chunk(pre_tag, cur_tag):
    if pre_tag == "O" or cur_tag == "O":
        return True

    if pre_tag[1:] != cur_tag[1:]:
        return True

    if cur_tag[0]=="B":
        return True

    return False

TP = {}
TotalPredict = {}
TotalLabel = {}

def report():

    out = sys.stdout

    for key in TotalPredict:
        if key not in TP:
            TP[key] = 0

    for key in TotalLabel:
        if key not in TP:
            TP[key] = 0

    for key in TP:
        if key == "O":
            continue
        p, r, f1 = calculate_metrics(TP[key], TotalPredict.get(key, 0), TotalLabel.get(key, 0))
        out.write('%17s: ' % key)
        out.write('precision: %6.2f%%; ' % (100. * p))
        out.write('recall: %6.2f%%; ' % (100. * r))
        out.write('FB1: %6.2f\n' % (100. * f1))

def evaluate(filename):
    label_unit = Unit()
    predict_unit = Unit()

    for line in open(filename, 'r'):
        line = line.strip()
        if len(line) == 0:
            continue
        word, label, predict = line.split()

        if word == "<S>":
            continue

        pre_label_end = end_of_chunk(label_unit.pre_tag, label)
        pre_predict_end = end_of_chunk(predict_unit.pre_tag, predict)

        if predict_unit.pre_tag == "" or label_unit.pre_tag=="":
            label_unit.pre_tag = label
            predict_unit.pre_tag = predict
            continue

        predict_tag = predict_unit.pre_tag
        if predict_tag != "O":
            predict_tag = predict_tag[2:]

        label_tag = label_unit.pre_tag
        if label_tag != "O":
            label_tag = label_unit.pre_tag[2:]

        if pre_label_end and pre_predict_end \
                and predict_unit.words == label_unit.words \
                and predict_unit.tags == label_unit.tags:
#            predict_unit.printInfo()
#            label_unit.printInfo()
            TP[predict_tag] = TP.get(predict_tag, 0) + 1

        if pre_label_end:
            TotalLabel[label_tag] = TotalLabel.get(label_tag, 0) + 1
            label_unit.words = word
            label_unit.tags = label
        else:
            label_unit.words = label_unit.words + word
            label_unit.tags = label_unit.tags + label

        if pre_predict_end:
            TotalPredict[predict_tag] = TotalPredict.get(predict_tag, 0) + 1
            predict_unit.words = word
            predict_unit.tags = predict
        else:
            predict_unit.words = predict_unit.words + word
            predict_unit.tags = predict_unit.tags + predict

        label_unit.pre_tag = label
        predict_unit.pre_tag = predict

        if word == "<E>":
            label_unit = Unit()
            predict_unit = Unit()

    report()

def main(argv):
    return evaluate(argv[1])

if __name__ == '__main__':
    sys.exit(main(sys.argv))
