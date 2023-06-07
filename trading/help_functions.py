from keras.utils import Sequence
import numpy as np


class CustomDataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X, self.y = X, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_X), np.array(batch_y)