import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# TODO add sentiment sources for given stock into same dataframe
class Dataset:

    FEATURES = ['open', 'close', 'high', 'low', 'volume']

    def __init__(self, data):
        # remove unnecessary columns
        self.data = data[Dataset.FEATURES]

        # partition data into train/test set
        self.partition()

    def partition(self, test_size=0.2, val_size=0.0):
        # default is train 80%, test 20%
        train_size = 1 - test_size - val_size

        # normalize data (squish values between 0 and 1)
        data = self.normalize(self.data, train_size)

        # convert multivariate time series data to a supervised learning format
        x, y = self.multivariate_to_supervised(data, lookback=30)
        train_samples = round(len(x) * train_size)

        # create train/val/test splits
        if val_size > 0.0:
            val_samples = round(self.data.shape[0] * val_size)
            self.train_val_test_split(x, y, train_samples, val_samples)

            print('Dataset: created %s training, %s validation, %s testing samples' %
                  (len(self.y_train), len(self.y_val), len(self.y_test)))
        else:
            self.x_train, self.x_test = np.array(x[:train_samples]), np.array(x[train_samples:])
            self.y_train, self.y_test = np.array(y[:train_samples]), np.array(y[train_samples:])

            print('Dataset: created %s training, %s testing samples' %
                  (self.y_train.shape[0], self.y_test.shape[0]))

    # TODO try StandardScaler
    def normalize(self, data, train_size):
        self.scaler = MinMaxScaler()

        # only fit on train
        train_samples = round(self.data.shape[0] * train_size)
        self.scaler.fit(data.values[:train_samples, :])

        data.loc[:, Dataset.FEATURES] = self.scaler.transform(data[Dataset.FEATURES])

        return data

    def multivariate_to_supervised(self, data, lookback):
        x, y = [], []

        for timepoint in range(data.shape[0] - lookback):
            x.append(data.values[timepoint:timepoint + lookback, :])
            y.append(data.values[timepoint + lookback, 0])

        return x, y

    def train_val_test_split(self, x, y, train_samples, val_samples):
        # train: start => train_samples
        # val: train_samples => train_samples + val_samples
        # test: train_samples + val_samples => end

        self.x_train = np.array(x[:train_samples])
        self.x_val = np.array(x[train_samples:train_samples + val_samples])
        self.x_test = np.array(x[train_samples + val_samples:])

        self.y_train = np.array(y[:train_samples])
        self.y_val = np.array(y[train_samples:train_samples + val_samples])
        self.y_test = np.array(y[train_samples + val_samples:])
