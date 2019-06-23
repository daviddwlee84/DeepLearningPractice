from embedding import Encoding
from dataset import setup_ner_data, get_total_word_set, train_test_trainable_to_numpy, from_ner_numpy_to_evaluable_format, get_ner_labels_from_file, raw_to_numpy, NER_LabelEncode
from basemodel import CRF, BiRNN_CRF
from evaluation import namedEntityEvaluation
from tqdm import tqdm
from constant import SUBMISSION, Function, Model, Embedding


# Functionality Switch
# 1. The 70% Train 30% Test model train
# 2. The 70% Train 30% Test model predict and evaluate
# 3. The Final Submit Predict model train
# 4. The Final Submit Predict model train predict
FUNC = Function(True, True, True, True)  # set False to skip steps
MODEL_TYPE = Model.BiRNN_CRF
ENCODE = Embedding.ONE_HOT
MODEL_NAME = ENCODE.value + '_' + MODEL_TYPE.value

# num_example = 60 sentences
BATCH_SIZE = 60  # determine the memory consumption (due to encoding)
EPOCH = 100


def train_test_experiment(train_set, test_set, encoder: Encoding, max_seq_len: int):
    (train_x, train_y, train_seq_len) = train_set
    (test_x, test_y, test_seq_len) = test_set

    num_examples, num_words = train_x.shape
    num_features = encoder.num_features
    num_tags = len(NER_LabelEncode)

    if MODEL_TYPE == Model.CRF:
        NER_train_test_model = CRF(num_words, num_features, num_tags,
                                   model_dir='model/ner_train_test/'+MODEL_NAME, model_name=MODEL_NAME)
    elif MODEL_TYPE == Model.BiRNN_CRF:
        NER_train_test_model = BiRNN_CRF(num_words, num_features, num_tags, max_seq_len,
                                         model_dir='model/ner_train_test/'+MODEL_NAME, model_name=MODEL_NAME)
    NER_train_test_model.build_model()

    if FUNC.Train_Test_Eval_train:
        print("Training 70% training data and test on the 30% training data")
        print(num_examples, num_words, num_features, num_tags)

        for epoch in range(EPOCH):
            print("Epoch:", epoch + 1)
            for batch_start in tqdm(range(0, num_examples, BATCH_SIZE)):
                if batch_start + BATCH_SIZE < num_examples:
                    batch_end = batch_start + BATCH_SIZE
                    echo = 0
                    save = 0
                else:
                    batch_end = num_examples
                    echo = 1
                    save = 1

                train_x_batch = train_x[batch_start:batch_end, :]
                train_x_batch = encoder.encode(train_x_batch)
                train_y_batch = train_y[batch_start:batch_end]
                train_seq_batch = train_seq_len[batch_start:batch_end]

                NER_train_test_model.train(
                    train_x_batch, train_y_batch, train_seq_batch, epoch=1, echo_per_epoch=echo, save_per_epoch=save)

    if FUNC.Train_Test_Eval_predict:
        print("Predict on 30% training data")

        num_predicts = test_x.shape[0]
        test_predict = []
        for batch_start in tqdm(range(0, num_predicts, BATCH_SIZE)):
            batch_end = batch_start + BATCH_SIZE if batch_start + \
                BATCH_SIZE < num_predicts else num_predicts

            test_x_batch = test_x[batch_start:batch_end, :]
            test_seq_batch = test_seq_len[batch_start:batch_end]

            test_x_encoded = encoder.encode(test_x_batch)
            test_batch_predict = NER_train_test_model.inference(
                test_x_encoded, test_seq_batch)
            test_predict.extend(test_batch_predict)

        return test_predict


def train_all_prediction(all_set, final_x, final_seq_len, encoder: Encoding, max_seq_len: int):
    (all_x, all_y, all_seq_len) = all_set

    num_examples, num_words = all_x.shape
    num_features = encoder.num_features
    num_tags = len(NER_LabelEncode)

    if MODEL_TYPE == Model.CRF:
        NER_all_model = CRF(num_words, num_features, num_tags,
                            model_dir='model/ner_all/'+MODEL_NAME, model_name=MODEL_NAME)
    elif MODEL_TYPE == Model.BiRNN_CRF:
        NER_all_model = BiRNN_CRF(num_words, num_features, num_tags, max_seq_len,
                                  model_dir='model/ner_all/'+MODEL_NAME, model_name=MODEL_NAME)
    NER_all_model.build_model()

    if FUNC.Final_Submit_train:
        print("Training on all the training data and predict on the final test data")
        print(num_examples, num_words, num_features, num_tags)

        for epoch in range(EPOCH):
            print("Epoch:", epoch)
            for batch_start in tqdm(range(0, num_examples, BATCH_SIZE)):
                if batch_start + BATCH_SIZE < num_examples:
                    batch_end = batch_start + BATCH_SIZE
                    echo = 0
                    save = 0
                else:
                    batch_end = num_examples
                    echo = 1
                    save = 1

                train_x_batch = all_x[batch_start:batch_end, :]
                train_x_batch = encoder.encode(train_x_batch)
                train_y_batch = all_y[batch_start:batch_end]
                train_seq_batch = all_seq_len[batch_start:batch_end]

                NER_all_model.train(
                    train_x_batch, train_y_batch, train_seq_batch, epoch=1, echo_per_epoch=echo, save_per_epoch=save)

    if FUNC.Final_Submit_predict:
        print("Predict on final test data")

        num_predicts = final_x.shape[0]
        final_predict = []
        for batch_start in tqdm(range(0, num_predicts, BATCH_SIZE)):
            batch_end = batch_start + BATCH_SIZE if batch_start + \
                BATCH_SIZE < num_predicts else num_predicts

            final_x_batch = final_x[batch_start:batch_end, :]
            final_seq_batch = final_seq_len[batch_start:batch_end]

            final_x_encoded = encoder.encode(final_x_batch)
            final_batch_predict = NER_all_model.inference(
                final_x_encoded, final_seq_batch)
            final_predict.extend(final_batch_predict)

        return final_predict


if __name__ == "__main__":
    train_data_list, test_data_list, train_all_list, final_raw_list = setup_ner_data()
    train_set, test_set, all_set = train_test_trainable_to_numpy(
        train_data_list, test_data_list, train_all_list, NER_LabelEncode)

    word_to_id, max_seq_len, word_set = get_total_word_set(
        train_all_list)
    encoder = Encoding(word_to_id, method=ENCODE.value)

    test_predict_filename = 'ner_test30percent_' + MODEL_NAME + '.txt'
    if FUNC.Train_Test_Eval_train or FUNC.Train_Test_Eval_predict:
        test_x, test_seq_len = test_set[0], test_set[2]
        test_predict = train_test_experiment(
            train_set, test_set, encoder, max_seq_len)
        if FUNC.Train_Test_Eval_predict:
            test30percent = from_ner_numpy_to_evaluable_format(
                test_x, test_predict, test_seq_len, word_to_id, NER_LabelEncode, test_predict_filename)
            # test from output file
            # test30percent = get_ner_labels_from_file(
            #     test_predict_filename, use_utf16_encoding=False)
            print("Evaluate on 30% training data")
            test_set_gold = [[label for (_, label) in sentence]
                             for sentence in test_data_list]
            namedEntityEvaluation(test30percent, test_set_gold)

    if FUNC.Final_Submit_train or FUNC.Final_Submit_predict:
        final_x, final_seq_len = raw_to_numpy(final_raw_list, train_all_list)
        final_predict = train_all_prediction(
            all_set, final_x, final_seq_len, encoder, max_seq_len)
        if FUNC.Final_Submit_predict:
            from_ner_numpy_to_evaluable_format(
                final_x, final_predict, final_seq_len, word_to_id, NER_LabelEncode, SUBMISSION.NER)
