from config.config import config
from data.tsx import TSX
from data.dataset import Dataset
from ml.lstm import LSTM

# load stock data from tsx
tsx = TSX(config)
data = tsx.get_history('A-CV', timeframe='1y')

# create dataset from data
dataset = Dataset(data)

# create lstm
lstm = LSTM(dataset)
lstm.train(epochs=20)
lstm.predict()

