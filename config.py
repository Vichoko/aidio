import pathlib
from os import makedirs as _makedirs

# General Settings
NUMBER_OF_CLASSES = 4
RANDOM_SEED = 69


# Configuration Settings
def makedirs(path):
    try:
        _makedirs(path)
    except FileExistsError:
        pass


AVAIL_MEDIA_TYPES = ['mp3', 'ogg', 'wav', 'flac', ]
FEATURE_EXTRACTOR_NUM_WORKERS = 4
DATA_LOADER_NUM_WORKERS = 4
CPU_NUM_WORKERS = 4

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

# MFCC
# params from 2011 Tsai et al.
MFCC_FFT_WINDOW = int(SR * 0.032)  # 32 ms window frame ~512
MFCC_HOP_LENGTH = int(SR * 0.010)  # 10 ms shifts ~128
MFCC_N_COEF = 20  # This is the most important param for fidelity, at 64 is much better but 20 was used on most
MFCC_N_MELS = N_MELS

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
TOP_DB_WINDOWED_MFCC = 38
MIN_INTERVAL_LEN_WINDOWED_MFCC = SR / 10
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
###########################
## trainer
# early stop
# EARLY_STOP_PATIENCE = 100
# EARLY_STOP_MONITOR = 'val_loss'

#########################################
####    MODELS
#########################################
# DUMMY MODE (2 classes; same train/val set)
DUMMY_EXAMPLES_PER_CLASS = 1

# GMM
GMM_PREDICT_BATCH_SIZE = 200
GMM_TRAIN_BATCH_SIZE = None  # used in data sampler of data loader; None is all posible
GMM_COMPONENT_NUMBER = 64
GMM_RANDOM_CROM_FRAME_LENGTH = 12 * 100  # 100 are 1 second; used on data loader; max is 1101
GMM_FIT_FRAME_LIMIT = 1000000 * GMM_RANDOM_CROM_FRAME_LENGTH  # 1000 are 1 second; 1000 * 60 is 1 minute; The frame limit is per-class as the fit is class-wise

# RNN 1D Classifier
RNN1D_BATCH_SIZE = 3000
RNN1D_LEARNING_RATE = 0.0001
RNN1D_WEIGHT_DECAY = 0.001
RNN1D_DOWNSAMPLER_OUT_CHANNELS = 256  # it has to be power of 2
RNN1D_DOWNSAMPLER_KERNEL_SIZE = 16
RNN1D_DOWNSAMPLER_STRIDE = 2
RNN1D_DOWNSAMPLER_DILATION = 1
RNN1D_HIDDEN_SIZE = RNN1D_DOWNSAMPLER_OUT_CHANNELS
RNN1D_LSTM_LAYERS = 1
RNN1D_BIDIRECTIONAL = False
RNN1D_DROPOUT_PROB = 0.0
RNN1D_MAX_SIZE = 4
RNN1D_FC1_INPUT_SIZE = RNN1D_MAX_SIZE * (RNN1D_HIDDEN_SIZE * 2 if RNN1D_BIDIRECTIONAL else RNN1D_HIDDEN_SIZE)
RNN1D_FC1_OUTPUT_SIZE = 2048
RNN1D_FC2_OUTPUT_SIZE = 512

# Conv1D Classifier
CONV1D_BATCH_SIZE = 2500
CONV1D_LEARNING_RATE = 0.001
CONV1D_WEIGHT_DECAY = 0.01
CONV1D_FEATURE_DIM = 256
CONV1D_STRIDE = 2
CONV1D_KERNEL_SIZE = 4
CONV1D_DILATION = 1
CONV1D_MAX_SIZE = 4

# FC
CONV1D_FC1_OUTPUT_DIM = 256
CONV1D_FC2_OUTPUT_DIM = 64

# WaveNet General
WAVEFORM_RANDOM_CROP_SEQUENCE_LENGTH = SR * 5  # native length is 176000 samples, i.e. 11 seconds
# Wavenet Vanilla Layers
WN_MAX_SIZE = 4
WAVENET_BATCH_SIZE = 20
WAVENET_LAYERS = 4
WAVENET_BLOCKS = 4
WAVENET_LEARNING_RATE = 0.001
WAVENET_WEIGHT_DECAY = 0.01
# FC
WAVENET_FC1_OUTPUT_DIM = 256
WAVENET_FC2_OUTPUT_DIM = 64
# Downsampling (for sentence encoder aproaches)
WAVENET_POOLING_KERNEL_SIZE = 32
WAVENET_POOLING_STRIDE = 32
# Dimensions
WAVENET_DILATION_CHANNELS = 32
WAVENET_RESIDUAL_CHANNELS = 32
WAVENET_SKIP_CHANNELS = 64
WAVENET_END_CHANNELS = 256
WAVENET_OUTPUT_LENGTH = 32
WAVENET_KERNEL_SIZE = 4
WAVENET_USE_AMSGRAD = False  # todo: integrate wavenet and wavenet + lstm
WAVENET_EPOCHS = 10000  # deprecated
WAVENET_CLASSES = 1  # Channels in (1 monoaural, 2 stereo)

# WaveNetTransformer
# WN
WNTF_MAX_SIZE = 4
WNTF_BATCH_SIZE = 40
WNTF_WAVENET_LAYERS = 3
WNTF_WAVENET_BLOCKS = 2
WNTF_LEARNING_RATE = 0.001
WNTF_WEIGHT_DECAY = 0.0000001
# Transformer
WNTF_TRANSFORMER_N_HEAD = 1
WNTF_TRANSFORMER_D_MODEL = 512
WNTF_TRANSFORMER_N_LAYERS = 2
WNTF_TRANSFORMER_DIM_FEEDFORWARD = 2048

# FC
WNTF_FC1_OUTPUT_DIM = 256
WNTF_FC2_OUTPUT_DIM = 64

# WaveNetLSTM
# WN
WNLSTM_MAX_SIZE = 4
WNLSTM_BATCH_SIZE = 24
WNLSTM_WAVENET_LAYERS = 4
WNLSTM_WAVENET_BLOCKS = 3
WNLSTM_LEARNING_RATE = 0.0001
WNLSTM_WEIGHT_DECAY = 0.0000001
# LSTM
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_POOL_TYPE = 'max'
LSTM_DROPOUT_PROB = 0.0
LSTM_BIDIRECTIONALITY = True
# FC
LSTM_FC1_OUTPUT_DIM = 256
LSTM_FC2_OUTPUT_DIM = 64

# Un-Used
#  ResNetV2
RESNET_V2_VERSION = 2
RESNET_V2_BATCH_SIZE = 20  # orig paper trained all networks with batch_size=128
RESNET_V2_EPOCHS = 200
RESNET_V2_DEPTH = 3 * 9 + 2
RESNET_V2_LR = 0.001
RESNET_V2_WEIGHT_DECAY = 0.01
RESNET_MIN_DIM = 29  # discovered by A/B testing

# Simpleconv
SIMPLECONV_BATCH_SIZE = 32
SIMPLECONV_EPOCHS = 64

# Simple1dConv
WAVEFORM_SAMPLE_RATE = SR
S1DCONV_EPOCHS = 1000
S1DCONV_BATCH_SIZE = 2
S1DCONV_NUM_WORKERS = 2
WAVEFORM_NUM_CHANNELS = 1  # can be 1 for mono or 2 for stereo; any other value will be stereo
S1DCONV_HIDDEN_BLOCKS = 1
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
