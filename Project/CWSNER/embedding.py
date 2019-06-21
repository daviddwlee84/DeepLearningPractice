import numpy as np


class Encoding:
    def __init__(self, word_to_id: dict, method: str = 'one-hot'):
        self.word_to_id = word_to_id
        self.num_features = None

        self.method = method
        if method == 'one-hot':
            self.num_features = len(self.word_to_id)

    def encode(self, numpy_data):
        # num_examples, num_words = numpy_data.shape

        if self.method == 'one-hot':
            # from word id numpy array (2-dim) to one-hot (3-dim)
            # return matrix shape: num_examples * num_words * num_features
            return np.eye(self.num_features, dtype=np.uint8)[numpy_data]

    def decode(self, encode_data):
        if self.method == 'one-hot':
            return np.argmax(encode_data, axis=2)


if __name__ == "__main__":
    numpy_data = np.array([[3, 2, 1, 4], [2, 1, 3, 0], [1, 2, 1, 1]])
    num_features = 5  # 0~4
    word_to_id = {i: i for i in range(num_features)}  # fake word_to_id
    print(numpy_data)

    print("Test one-hot")
    one_hot_encoder = Encoding(word_to_id, method='one-hot')
    one_hot_data = one_hot_encoder.encode(numpy_data)
    print(one_hot_data)
    one_hot_back_data = one_hot_encoder.decode(one_hot_data)
    print(one_hot_back_data)
