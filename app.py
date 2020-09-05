from config.config import config
from data.tsx import TSX
from data.dataset import Dataset
from ml.lstm import LSTM

# load stock data from tsx
tsx = TSX(config)
data = tsx.get_history('A-CV', timeframe='5y')

if data is None:
    exit()

# create dataset from data
dataset = Dataset(data)

# create lstm
lstm = LSTM(dataset)
#lstm.train(epochs=20, name='basic_lstm')
lstm.predict('basic_lstm', plot=True)
