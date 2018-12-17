import numpy as np
import matplotlib.pyplot as plt

TIMESTEPS = 10            # Training sequence length of RNN
TRAINING_EXAMPLES = 10000 # Training data amount
TESTING_EXAMPLES = 1000   # Testing data amount
SAMPLE_GAP = 0.01         # Sampling interval


def _seperate_data(seq):
    """Generate Data Sequence"""
    X = []
    y = []
    # Use sequence of [i, i+TIMESTEPS) to predict i+TIMESTEPS
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  

def generate_data(gen_func):
    """Generate Training and testing set"""
    test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    train_X, train_y = _seperate_data(gen_func(np.linspace(
        0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    test_X, test_y = _seperate_data(gen_func(np.linspace(
        test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    return train_X, train_y, test_X, test_y

if __name__ == "__main__":
    train_X, train_y, test_X, test_y = generate_data(np.sin)
    train_X, train_y, test_X, test_y = generate_data(np.cos)
    