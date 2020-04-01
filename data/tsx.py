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
            print('An error occured while initializing data.TSX()')
            exit()

    def load_symbols(self):
        try:
            self.symbols = self.api.internationalSymbolsDF(exchange='tsx')
            print('TSX(): loaded %s symbols' % self.symbols.shape[0])
        except Exception:
            print('An error occured while loading TSX symbols')
        
        
