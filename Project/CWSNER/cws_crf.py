from embedding import Encoding
from dataset import setup_cws_data, get_total_word_set, train_test_trainable_to_numpy, from_trainable_to_cws_list, from_cws_numpy_to_evaluable_format, raw_to_numpy, CWS_LabelEncode
from basemodel import CRF
from evaluation import wordSegmentEvaluaiton
from tqdm import tqdm
from constant import SUBMISSION

# num_example = 60 sentences
BATCH_SIZE = 60
EPOCH = 1000


def train_test_experiment(train_set, test_set, encoder: Encoding):
    (train_x, train_y, train_seq_len) = train_set
    (test_x, test_y, test_seq_len) = test_set

    num_examples, num_words = train_x.shape
    num_features = encoder.num_features
    num_tags = len(CWS_LabelEncode)

    print("Training 70% training data and test on the 30% training data")
    print(num_examples, num_words, num_features, num_tags)

    CRF_train_test_model = CRF(num_words, num_features, num_tags,
                               model_dir='model/cws_train_test', model_name='one-hot_crf')
    CRF_train_test_model.build_model()

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

            CRF_train_test_model.train(
                train_x_batch, train_y_batch, train_seq_batch, epoch=1, echo_per_epoch=echo, save_per_epoch=save)

    print("Predict on 30% training data")

    num_predicts = test_x.shape[0]
    test_predict = []
    for batch_start in tqdm(range(0, num_predicts, BATCH_SIZE)):
        batch_end = batch_start + BATCH_SIZE if batch_start + \
            BATCH_SIZE < num_predicts else num_predicts

        test_x_batch = test_x[batch_start:batch_end, :]
        test_seq_batch = test_seq_len[batch_start:batch_end]

        test_x_encoded = encoder.encode(test_x_batch)
        test_batch_predict = CRF_train_test_model.inference(
            test_x_encoded, test_seq_batch)
        test_predict.extend(test_batch_predict)

    return test_predict


def train_all_prediction(all_set, final_x, final_seq_len, encoder: Encoding):
    (all_x, all_y, all_seq_len) = all_set

    num_examples, num_words = all_x.shape
    num_features = encoder.num_features
    num_tags = len(CWS_LabelEncode)

    print("Training on all the training data and predict on the final test data")
    print(num_examples, num_words, num_features, num_tags)

    CRF_all_model = CRF(num_words, num_features, num_tags,
                        model_dir='model/cws_all', model_name='one-hot_crf')
    CRF_all_model.build_model()

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

            CRF_all_model.train(
                train_x_batch, train_y_batch, train_seq_batch, epoch=1, echo_per_epoch=echo, save_per_epoch=save)

    print("Predict on final test data")

    num_predicts = final_x.shape[0]
    final_predict = []
    for batch_start in tqdm(range(0, num_predicts, BATCH_SIZE)):
        batch_end = batch_start + BATCH_SIZE if batch_start + \
            BATCH_SIZE < num_predicts else num_predicts

        final_x_batch = final_x[batch_start:batch_end, :]
        final_seq_batch = final_seq_len[batch_start:batch_end]

        final_x_encoded = encoder.encode(final_x_batch)
        final_batch_predict = CRF_all_model.inference(
            final_x_encoded, final_seq_batch)
        final_predict.extend(final_batch_predict)

    return final_predict


if __name__ == "__main__":
    train_data_list, test_data_list, train_all_list, final_raw_list = setup_cws_data()
    train_set, test_set, all_set = train_test_trainable_to_numpy(
        train_data_list, test_data_list, train_all_list, CWS_LabelEncode, fixed_max_seq_len=165)

    word_to_id, max_seq_len, word_set = get_total_word_set(
        train_all_list, fixed_max_seq_len=165)
    encoder = Encoding(word_to_id, method='one-hot')

    test_x, test_seq_len = test_set[0], test_set[2]
    test_predict = train_test_experiment(train_set, test_set, encoder)
    test30percent = from_cws_numpy_to_evaluable_format(
        test_x, test_predict, test_seq_len, word_to_id, CWS_LabelEncode, 'cws_test30percent.txt')
    print("Evaluate on 30% training data")
    test_set_gold = from_trainable_to_cws_list(test_data_list)
    with open('cws_test30percent.txt', 'r') as f:
        test30percent = f.readlines()
    wordSegmentEvaluaiton(test30percent, test_set_gold)

    final_x, final_seq_len = raw_to_numpy(final_raw_list, train_all_list)
    final_predict = train_all_prediction(
        all_set, final_x, final_seq_len, encoder)
    from_cws_numpy_to_evaluable_format(
        final_x, final_predict, final_seq_len, word_to_id, CWS_LabelEncode, SUBMISSION.CWS)
