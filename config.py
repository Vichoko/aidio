import pathlib
from os import makedirs as _makedirs


def makedirs(path):
    try:
        _makedirs(path)
    except FileExistsError:
        pass


AVAIL_MEDIA_TYPES = ['mp3', 'ogg', 'wav', 'flac', ]
CPU_WORKERS = 4

SOURCE_DATA_PATH = pathlib.Path('C:\\Users\\Vichoko\\Music\\in')
RAW_DATA_PATH = pathlib.Path('./data/raw/')
FEATURES_DATA_PATH = pathlib.Path('./data/features/')
DIGEST_DATA_PATH = pathlib.Path('./data/digest/')
MODELS_DATA_PATH = pathlib.Path('./data/models/')

makedirs(RAW_DATA_PATH)
makedirs(FEATURES_DATA_PATH)
makedirs(DIGEST_DATA_PATH)

# -- Audio processing parameters --#
SR = 16000

# Mel Frequency Spectrogram
N_FFT = 2048  # length of the FFT window 20 to 30 ms
HOP_LENGTH = 512  # number of samples between successive frames.
POWER = 2  # Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.

# MEL FILTER PARAMS
N_MELS = 128  # number of Mel bands to generate
FMIN = 0  # lowest frequency (in Hz)
FMAX = None  # Highest frequency (in Hz)


# -- sINGING vOICE dETECTION --#
VOICE_DETECTION_PATH = '/home/voyanedel/data/code/ismir2018-revisiting-svd/'
VOICE_DETECTION_MODEL_NAME = 'june2019'
RNN_INPUT_SIZE_VOICE_ACTIVATION = 218  # 7sec/(256/16000)
RNN_OVERLAP_VOICE_ACTIVATION = 10

# -- HPSS FEATURE EXTRACTION --
SR_HPSS = 16000
N_MELS_HPSS = 40

N_FFT_HPSS_1 = 4096
N_HOP_HPSS_1 = 2048
N_FFT_HPSS_2 = 512
N_HOP_HPSS_2 = 256