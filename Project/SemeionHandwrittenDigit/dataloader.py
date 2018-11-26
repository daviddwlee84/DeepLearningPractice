import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Train and Test
class ImageData:
    def __init__(self, data, label):
        self.__batch_pointer = 0
        self.images = data
        self.labels = label
        self.num_examples, self.num_features = np.shape(data)
        self.__one_hot_len = len(self.labels[0])
    
    # Return a batch of data and label
    def next_batch(self, batch_size):
        if self.__batch_pointer + batch_size <= self.num_examples:
            returnData = self.images[self.__batch_pointer:self.__batch_pointer + batch_size, :]
            returnLabel = self.labels[self.__batch_pointer:self.__batch_pointer + batch_size, :]
        else:
            exceednum = (self.__batch_pointer + batch_size) % self.num_examples

            returnData = np.zeros((batch_size, self.num_features))
            returnData[:-exceednum] = self.images[self.__batch_pointer:, :]
            returnData[-exceednum:] = self.images[:exceednum, :]

            returnLabel = np.zeros((batch_size, self.__one_hot_len))
            returnLabel[:-exceednum] = self.labels[self.__batch_pointer:, :]
            returnLabel[-exceednum:] = self.labels[:exceednum, :]

        # Update batch pointer
        self.__batch_pointer = (self.__batch_pointer + batch_size) % self.num_examples
        return returnData, returnLabel

# Data set
# Now only support one-hot label
class ImageDataSet:
    def __init__(self, data, label, test_set_ratio=0.3, random_seed=87):
        data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=test_set_ratio, random_state=random_seed)
        self.train = ImageData(data_train, label_train)
        self.test = ImageData(data_test, label_test)

def loadSemeionData(path, random_state=87):
    rowdata = pd.read_csv(path, sep=' ', header=None)
    rowdata = rowdata.iloc[:, :-1] # Remove last column (NaN)
    rowdata = rowdata.sample(frac=1, random_state=random_state).reset_index(drop=True) # Shuffle
    data = rowdata.iloc[:, :-10]
    label = rowdata.iloc[:, -10:]
    return np.array(data), np.array(label)

def getSemeionData(test=False):
    data, label = loadSemeionData('semeion.data', random_state=87)

    if test:
        return ImageDataSet(data[:15, :], label[:15, :], test_set_ratio=0.3)
    else:
        return ImageDataSet(data, label, test_set_ratio=0.2, random_seed=87)

def main():
    test = getSemeionData(test=True)
    print(test.train.next_batch(10))
    print(test.train.next_batch(10))
    print(test.train.num_examples)
    print(test.train.num_features)
    print(test.train.images[0])

if __name__ == "__main__":
    main()