class RAW_DATA:
    CWS = 'data/Train_utf16.seg'
    NER = 'data/Train_utf16.ner'
    CWS_test = 'data/Test_utf16.seg'
    NER_test = 'data/Test_utf16.ner'


# For 7:3 training data and test data
# Transfer the test data into raw article for test
# And the rest of the data as train data
class TRAIN_TEST:
    CWS_train = 'train_test/cws_train.txt'  # the training data (70%data)
    CWS_train_pkl = 'train_test/cws_train.pkl'  # the trainable format (70% data)
    CWS_train_all_pkl = 'train_test/cws_train_all.pkl' # the trainable format (all data)
    CWS_test = 'train_test/cws_test.txt'  # the test data (with label) (30% data)
    CWS_test_pkl = 'train_test/cws_test.pkl'  # the testable format (30% data)
    CWS_final_pkl = 'train_test/cws_final.pkl' # the testable format (final test data)

    NER_train = 'train_test/ner_train.txt'  # the training data (70%data)
    NER_train_all_pkl = 'train_test/ner_train_all.pkl' # the trainable format (all data)
    NER_test = 'train_test/ner_test.txt'  # the test data (with label) (30% data)
    NER_final_pkl = 'train_test/ner_final.pkl' # the testable format (final test data) 


class SUBMISSION:
    CWS = '李大為-1701210963.cws'
    NER = '李大為-1701210963.ner'

