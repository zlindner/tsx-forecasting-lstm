from config.config import config
from data.tsx import TSX

tsx = TSX(config)

print(tsx.get_history('A-CV'))