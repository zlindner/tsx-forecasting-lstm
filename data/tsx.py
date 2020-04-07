import pyEX
import os
import pandas as pd


class TSX:

    def __init__(self, config):
        self.init_api(config)  # TODO move to iex.py or something when implementing U.S exchanges
        self.load_symbols()

    def init_api(self, config):
        try:
            self.api = pyEX.Client(api_token=config['api_key'], version=config['api_version'])
            print('TSX: initialized')
        except Exception:
            print('TSX: an error occured while initializing')
            exit()

    def load_symbols(self):
        if os.path.exists('data/tsx/symbols.csv'):
            print('TSX: loading cached symbols from data/tsx/symbols.csv')
            self.symbols = pd.read_csv('data/tsx/symbols.csv', index_col=0)
        else:
            print('TSX: loading symbols from iex')

            try:
                self.symbols = self.api.internationalSymbolsDF(exchange='tsx')
                self.symbols.to_csv('data/tsx/symbols.csv')
            except Exception:
                print('TSX: an error occured while loading symbols')
                return

        print('TSX: loaded %s symbols' % self.symbols.shape[0])

    def get_history(self, symbol, timeframe):
        if symbol not in self.symbols['name']:
            print('TSX: invalid symbol')
            return

        filename = 'data/tsx/tsx-' + symbol + '-' + timeframe + '.csv'

        if os.path.exists(filename):
            print('TSX: loading cached history from %s' % filename)
            history = pd.read_csv(filename, index_col=0)
            return history
        try:
            print('TSX: loading history(%s) for %s from iex' % (timeframe, symbol))
            history = self.api.chartDF(symbol, timeframe)
            history.to_csv(filename)

            return history
        except Exception as err:
            print('TSX: an error occured while retrieving historical data')
            print(err)

            return None
