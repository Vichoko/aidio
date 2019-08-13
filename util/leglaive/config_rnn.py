'''
Config for LSTM model 
'''
import pathlib
# predictions
from os import makedirs

DEFAULT_MODEL = 'june2019'
PREDICTIONS_DIR = pathlib.Path('./data/cache/predictions')
MEL_CACHE_DIR = pathlib.Path('./data/cache/mel_cache')
WEIGHTS_DIR = pathlib.Path('./data/models/leglaive/')

try:
    makedirs(PREDICTIONS_DIR)
except Exception:
    pass
try:
    makedirs(MEL_CACHE_DIR)
except Exception:
    pass


# -- Audio processing parameters --#


N_MFCC = 40

# -- model parameters --#
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
INPUT_SIZE = 80  # 40 harmonic + 40 percussive
NUM_EPOCHS = 100
THRESHOLD = 0.5
