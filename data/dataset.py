import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# TODO add sentiment sources for given stock into same dataframe
class Dataset:

    FEATURES = ['open', 'close', 'high', 'low', 'volume']

    def __init__(self, data):
        # remove unnecessary columns
        self.data = data[Dataset.FEATURES]

        # partition data into train/test set
        self.partition()

    def partition(self, test_size=0.2):
        # convert multivariate time series data to a supervised learning format
        data = self.multivariate_to_supervised(self.data)
        print(data.head())

        # default is train 80%, test 20%
        train_samples = round(data.shape[0] * (1 - test_size))
        # normalize data (squish values between 0 and 1)
        data = self.normalize(data, train_samples)
        print(data.shape)
        '''
        train = data[:train_samples, :]
        test = data[train_samples:, :]

        # split into inputs and outputs
        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        
        return x_train, y_train, x_test, y_test'''

    # TODO try StandardScaler
    def normalize(self, data, train_samples):
        self.scaler = MinMaxScaler()

        # only fit on train
        self.scaler.fit(data.values[:train_samples, :])
        return self.scaler.transform(data.values)

    def multivariate_to_supervised(self, data, prev_steps=1, forecast_steps=1):
        '''
        Transforms a DataFrame containing time series data into a DataFrame
        containing data suitable for use as a supervised learning problem.
        
        Derived from code originally found at 
        https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
        
        :param data: dataframe containing columns of time series values
        :param prev_steps: the number of previous steps that will be included in the output 
                            DataFrame corresponding to each input column
        :param forecast_steps: the number of forecast steps that will be included in the output 
                            DataFrame corresponding to each input column
        :return dataframe containing original columns, renamed <orig_name>(t), as well as columns
                            for previous steps, <orig_name>(t-1) ... <orig_name>(t-n) and 
                            columns for forecast steps, <orig_name>(t+1) ... <orig_name>(t+n)
        '''

        cols, names = list(), list()

        # input sequence (t-n, ... t-1)
        for i in range(prev_steps, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col_name, i)) for col_name in Dataset.FEATURES]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, forecast_steps):
            cols.append(data.shift(-i))

            if i == 0:
                names += [('%s(t)' % col_name) for col_name in Dataset.FEATURES]
            else:
                names += [('%s(t+%d)' % (col_name, i)) for col_name in Dataset.FEATURES]

        # aggregate columns
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)

        return agg