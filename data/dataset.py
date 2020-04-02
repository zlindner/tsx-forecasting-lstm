from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Dataset:

    FEATURES = ['open', 'close', 'high', 'low', 'volume']

    def __init__(self, data):
        self.create(data)

    def create(self, data):
        # remove unnecessary columns
        self.data = data[Dataset.FEATURES]

        # partition data into train/test set
        self.x_train, self.y_train, self.x_test, self.y_test = self.partition()

    def partition(self, test_size=0.2):
        # default is train 80%, test 20%
        train_size = round(self.data.shape[0] * (1 - test_size))
        train = self.data.values[:train_size, :]
        test = self.data.values[train_size:, :]

        # normalize data using a min-max scaler
        train, test = self.normalize(train, test)

        # split into inputs and outputs
        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        
        return x_train, y_train, x_test, y_test

    def normalize(self, train, test):
        self.scaler = MinMaxScaler()
        
        # only fit on train
        train = self.scaler.fit_transform(train) 
        test = self.scaler.transform(test)

        return train, test
