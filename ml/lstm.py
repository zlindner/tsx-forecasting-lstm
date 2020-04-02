import tensorflow.keras as tf
import numpy as np

class LSTM:

    def __init__(self, dataset):
        self.dataset = dataset
        self.create()

    def create(self):
        train_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2])

        self.model = tf.models.Sequential()

        self.model.add(tf.layers.LSTM(128, input_shape=train_shape, return_sequences=True))
        self.model.add(tf.layers.LSTM(64, input_shape=train_shape))
        self.model.add(tf.layers.Dense(16, activation='relu'))
        self.model.add(tf.layers.Dense(1, activation='linear'))

        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')

    def train(self, epochs):
        self.model.fit(self.dataset.x_train, self.dataset.y_train, epochs=epochs, batch_size=32, shuffle=False)

    def predict(self):
        x_test = self.dataset.x_test
        y_test = self.model.predict(x_test)

        x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
        
        # invert scaling for forecast
        inv_yhat = np.concatenate((y_test, x_test[:, 1:]), axis=1)
        inv_yhat = self.dataset.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        
        # invert scaling for actual
        #test_y = test_y.reshape((len(test_y), 1))
        #inv_y = np.concatenate((test_y, x_test[:, 1:]), axis=1)
        #inv_y = self.dataset.scaler.inverse_transform(inv_y)
        #inv_y = inv_y[:,0]
        
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
