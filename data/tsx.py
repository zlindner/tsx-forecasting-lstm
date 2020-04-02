import pyEX

class TSX:

    def __init__(self, config):
        self.init_api(config) # TODO move to iex.py or something when implementing U.S exchanges
        self.load_symbols()

    def init_api(self, config):
        try:
            self.api = pyEX.Client(api_token=config['api_key'], version=config['api_version'])
            print('TSX(): initialized')
        except Exception:
            print('TSX(): an error occured while initializing')
            exit()

    def load_symbols(self):
        try:
            self.symbols = self.api.internationalSymbolsDF(exchange='tsx')
            print('TSX(): loaded %s symbols' % self.symbols.shape[0])
        except Exception:
            print('TSX(): an error occured while loading symbols')

    def get_history(self, symbol, timeframe='1m'):
        if symbol not in self.symbols['name']:
            print('TSX(): invalid symbol')
            return

        print('TSX(): loading history (%s) for %s' % (timeframe, symbol))

        try:
            return self.api.chartDF(symbol, timeframe)
        except Exception:
            print('TSX(): an error occured while retrieving historical data')
            
        return None
        
        
