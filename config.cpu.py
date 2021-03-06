import pathlib
from os import makedirs as _makedirs


def makedirs(path):
    try:
        _makedirs(path)
    except FileExistsError:
        pass


AVAIL_MEDIA_TYPES = ['mp3', 'ogg', 'wav', 'flac', ]
NUM_WORKERS = 4

SOURCE_DATA_PATH = pathlib.Path('C:\\Users\\Vichoko\\Music\\in')
RAW_DATA_PATH = pathlib.Path('./data/raw/')
FEATURES_DATA_PATH = pathlib.Path('./data/features/')
DIGEST_DATA_PATH = pathlib.Path('./data/digest/')
MODELS_DATA_PATH = pathlib.Path('./data/models/')
OTHER_DATA_PATH = pathlib.Path('./data/other/')

makedirs(RAW_DATA_PATH)
makedirs(FEATURES_DATA_PATH)
makedirs(DIGEST_DATA_PATH)
makedirs(MODELS_DATA_PATH)
makedirs(OTHER_DATA_PATH)

#########################################
####    FEATURES
#########################################

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

# -- WINDOW MFCC --
TOP_DB_WINDOWED_MFCC = 80
MIN_INTERVAL_LEN_WINDOWED_MFCC = SR
WINDOW_LEN_WINDOWED_MFCC = SR
WINDOW_HOP_WINDOWED_MFCC = int(SR / 3)

# Magnitude Phase STFT
MAGPHASE_SAMPLE_RATE = 8192
MAGPHASE_WINDOW_SIZE = 1024
MAGPHASE_HOP_LENGTH = 768
MAGPHASE_PATCH_SIZE = 128

# OpenUnmixPytorch

OUNMIX_SAMPLE_RATE = 44100
OUNMIX_NITER = 1
OUNMIX_ALPHA = 1
OUNMIX_TARGETS = ['vocals']
OUNMIX_SOFTMAX = False
OUNMIX_RESIDUAL_MODEL = False
OUNMIX_MODEL = 'umxhq'

#########################################
####    MODELS
#########################################

#  ResNetV2
RESNET_V2_VERSION = 2
RESNET_V2_BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128
RESNET_V2_EPOCHS = 200
RESNET_V2_DEPTH = 3 * 9 + 2
RESNET_MIN_DIM = 29  # discovered by A/B testing

# Simpleconv
SIMPLECONV_BATCH_SIZE = 32
SIMPLECONV_EPOCHS = 64

# Simple1dConv
WAVEFORM_SAMPLE_RATE = SR
S1DCONV_EPOCHS = 1000
S1DCONV_BATCH_SIZE = 2
S1DCONV_NUM_WORKERS = 2
WAVEFORM_SEQUENCE_LENGTH = SR * 10
WAVEFORM_NUM_CHANNELS = 1  # can be 1 for mono or 2 for stereo; any other value will be stereo
S1DCONV_HIDDEN_BLOCKS = 1

# WaveNet

WAVENET_LAYERS = 1
WAVENET_BLOCKS = 2
WAVENET_DILATION_CHANNELS = 16
WAVENET_RESIDUAL_CHANNELS = 16
WAVENET_SKIP_CHANNELS = 32
WAVENET_END_CHANNELS = 64
WAVENET_CLASSES = 1
WAVENET_OUTPUT_LENGTH = 32
WAVENET_KERNEL_SIZE = 2
WAVENET_EPOCHS = 1000
WAVENET_BATCH_SIZE = 1
WAVENET_POOLING_KERNEL_SIZE = 100
WAVENET_POOLING_STRIDE = 50

# Infersent
# BiLSTM w& Max Pooling ecoding
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
LSTM_POOL_TYPE = 'max'
LSTM_DROPOUT_PROB = 0.0

# ADiSAN
ADISAN_BATCH_SIZE = 64
ADISAN_EPOCHS = 200
ADISAN_DROPOUT_KEEP_PROB = 0.7
ADISAN_HIDDEN_UNITS = 300
ADISAN_LR = 0.5  # initial learning rate
ADISAN_DECAY = 0.9  # summary decay ema
ADISAN_VAR_DECAY = 0.999  # learing rate ema
ADISAN_WEIGHT_DECAY_FACTOR = 1e-4  # weigh decay factor / l2 decay factor
ADISAN_OPTIMIZER = 'adadelta'  # [adadelta|adam]
ADISAN_MAX_STEPS = 120000  # total (batch) steps to be performed during training
# dudoso
ADISAN_LOG_PERIOD = 500  # save tf summary period
ADISAN_SAVE_PERIOD = 3000
ADISAN_EVAL_PERIOD = 1000
ADISAN_GPU_MEM = 0.96  # GPU memoty ratio

# deprecated
ADISAN_LOAD_PATH = OTHER_DATA_PATH / 'load_model'  # deprecated; for test mode; specify which pre-trianed model to be load
ADISAN_LOAD_MODEL = False  # force load model from chkp on instance
ADISAN_LOAD_STEP = None
ADISAN_SUMMARY_PATH = OTHER_DATA_PATH / 'summary'

ADISAN_CHECKPOINT_PATH = OTHER_DATA_PATH / 'checkpoints'
ADISAN_ANSWER_DIR = OTHER_DATA_PATH / 'answers'

ADISAN_LOG_DIR = OTHER_DATA_PATH / 'logs'
ADISAN_STANDBY_LOG_DIR = OTHER_DATA_PATH / 'standby_log'

makedirs(ADISAN_LOAD_PATH)
makedirs(ADISAN_SUMMARY_PATH)
makedirs(ADISAN_CHECKPOINT_PATH)
makedirs(ADISAN_ANSWER_DIR)
makedirs(ADISAN_LOG_DIR)
makedirs(ADISAN_STANDBY_LOG_DIR)
