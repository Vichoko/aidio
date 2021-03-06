import math
import os
import pickle
import typing
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from sklearn import mixture
from sklearn.exceptions import NotFittedError
from torch import nn as nn
from torch.nn import functional as F

from config import WAVENET_LAYERS, WAVENET_BLOCKS, WAVENET_DILATION_CHANNELS, WAVENET_RESIDUAL_CHANNELS, \
    WAVENET_SKIP_CHANNELS, \
    WAVENET_END_CHANNELS, WAVENET_CLASSES, WAVENET_OUTPUT_LENGTH, WAVENET_KERNEL_SIZE, LSTM_HIDDEN_SIZE, \
    LSTM_NUM_LAYERS, LSTM_DROPOUT_PROB, WAVEFORM_RANDOM_CROP_SEQUENCE_LENGTH, \
    WNTF_TRANSFORMER_D_MODEL, WNTF_TRANSFORMER_N_HEAD, WNTF_TRANSFORMER_N_LAYERS, FEATURES_DATA_PATH, \
    GMM_COMPONENT_NUMBER, \
    GMM_FIT_FRAME_LIMIT, WNTF_WAVENET_LAYERS, WNTF_WAVENET_BLOCKS, \
    WNLSTM_WAVENET_LAYERS, WNLSTM_WAVENET_BLOCKS, CPU_NUM_WORKERS, CONV1D_FEATURE_DIM, RNN1D_DOWNSAMPLER_OUT_CHANNELS, \
    RNN1D_DROPOUT_PROB, RNN1D_HIDDEN_SIZE, RNN1D_LSTM_LAYERS, RNN1D_BIDIRECTIONAL, RNN1D_DOWNSAMPLER_STRIDE, \
    RNN1D_DOWNSAMPLER_KERNEL_SIZE, RNN1D_DOWNSAMPLER_DILATION, CONV1D_KERNEL_SIZE, CONV1D_STRIDE, CONV1D_DILATION, \
    LSTM_FC1_OUTPUT_DIM, LSTM_FC2_OUTPUT_DIM, CONV1D_FC1_OUTPUT_DIM, CONV1D_FC2_OUTPUT_DIM, WNTF_FC1_OUTPUT_DIM, \
    WNTF_FC2_OUTPUT_DIM, WNTF_TRANSFORMER_DIM_FEEDFORWARD, LSTM_BIDIRECTIONALITY, WAVENET_FC1_OUTPUT_DIM, \
    WAVENET_FC2_OUTPUT_DIM, RNN1D_MAX_SIZE, RNN1D_FC1_INPUT_SIZE, RNN1D_FC1_OUTPUT_SIZE, RNN1D_FC2_OUTPUT_SIZE, \
    CONV1D_MAX_SIZE, WNTF_MAX_SIZE, WNLSTM_MAX_SIZE, WN_MAX_SIZE
from util.wavenet.wavenet_model import WaveNetModel


class GMMClassifier(nn.Module):

    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes, n_components=GMM_COMPONENT_NUMBER, fit_frame_limit=GMM_FIT_FRAME_LIMIT):
        super(GMMClassifier, self).__init__()
        self.fit_frame_limit = fit_frame_limit
        self.gmm_list = []
        for _ in range(num_classes):
            # one gmm per singer as stated in Tsai; Fujihara; Mesaros et. al works on SID
            self.gmm_list.append(
                mixture.GaussianMixture(n_components=n_components)
            )

    def forward(self, x):
        """
        Do a forward pass obtaining de score for each element.
        If all elements are of the same length (same n_frames), the process is optimized by doing a flatten.
        :param x: torch.Tensor MFCC of a track with shape (n_element, n_features, n_frames, )
        :return: torch.Tensor The prediction for each track calculated as:
            Singer_id = arg max_i [1/T sum^T_t=1 [log p(X_t / P_i)]]

            where t is time frame and
                i is the singer GMM
                shape is (batch_size, n_classes)
        """
        # asume that all the samples has equal frame number
        x = x.permute(0, 2, 1)  # shape (batch_size, n_features, n_frames) to (batch_size, n_frames, n_features)
        batch_size = x.size(0)
        n_frames = x.size(1)
        n_features = x.size(2)
        x = x.reshape(batch_size * n_frames,
                      n_features)  # flatten 2 first dimensions into one (n_total_frames, n_features)
        x = self.forward_score(x)  # output shape is (n_classes, n_total_frames)
        n_classes = x.size(0)
        x = x.view(n_classes, batch_size, n_frames)  # un_flatten to recover elementwise frames
        x = torch.sum(x, dim=2)  # sum the probability of each element's frames; shape is (n_classes, batch_size)
        x = x.permute(1, 0)  # swap axes to match signature
        return x

    def fit(self, x, y):
        """
        Fit a sequence of frames of the same class into one of the
        gmm.
        :param x: Training data (batch_element, n_features, n_frames)
        :param y: class id integer singleton tensor
        :return:
        """
        # sklearn GMM expects (n, n_features)
        debug = True
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, 20)
        # print('Debug: y = {}'.format(y)) if debug else None
        # print('Debug: x = {}'.format(x)) if debug else None
        # print('Debug: gmm_list = {}'.format(self.gmm_list)) if debug else None
        print('info: Fitting GMM...')
        data = x[:self.fit_frame_limit, :]
        print('Debug: Training data have shape {}'.format(data.shape)) if debug else None
        self.gmm_list[y[0].item()].fit(data)
        print('info: Done!')

    def save_gmm(self, gmm_idx: int, path):
        """
        Save indexed GMM on storage.
        :param gmm_idx: Index of the GMM corresponding to the nominal label trained to predict.
        :param path: Absolute path to the storage file to open.
        :return:
        """
        assert not os.path.isfile(path), 'error: Saving GMM instance noted that {} already exists'.format(path)
        pickle.dump(self.gmm_list[gmm_idx], open(path, 'wb'))

    def load_gmm(self, gmm_idx: int, path):
        """
        Loaded GMM from storage to this instance given index.
        May raise FileNotFoundException if path doesn't exists.
        :param gmm_idx: Index of the GMM representing the corresponding nominal label trained to predict.
        :param path: Absolute path to the storage file to open.
        :return:
        """
        self.gmm_list[gmm_idx] = pickle.load(open(path, 'rb'))

    def forward_score(self, x):
        """
        :param x: MFCC of a track with shape (frames, coefficients, )
        :return: The Log Likelihood for each track and frame tested on every GMM (one per singer / class) as:
            log likelihood = log p(X_t / P_i)]

            where t is time frame and
                i is the singer GMM

            with shape: (n_classes/n_gmm, n_frames)
        """

        def get_scores_from_gmm(gmm):
            try:
                return torch.from_numpy(gmm.score_samples(x))  # output is a (n_frames)
            except NotFittedError:
                return torch.zeros(n_frames, dtype=torch.double) + float('-inf')

        n_frames = x.size(0)
        # n_features = x.size(1)
        with ThreadPoolExecutor(CPU_NUM_WORKERS) as e:
            framewise_scores = e.map(get_scores_from_gmm, self.gmm_list)
        y = torch.stack(list(framewise_scores))  # reshape to tensor (n_classes, n_frames)
        return y


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Conv1DClassifier(nn.Module):
    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes):
        super(Conv1DClassifier, self).__init__()
        # first encoder
        # neural audio embeddings
        # captures local representations through convolutions
        # note: x.shape is (bs, 1, ~80000)
        n_layers = int(math.log2(CONV1D_FEATURE_DIM))
        self.conv_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=2 ** layer_idx,
                    out_channels=2 ** (layer_idx + 1),
                    kernel_size=CONV1D_KERNEL_SIZE,
                    stride=CONV1D_STRIDE,
                    dilation=CONV1D_DILATION,
                )
            )

        self.max_pool = nn.AdaptiveAvgPool1d(CONV1D_MAX_SIZE)
        self.avg_pool = nn.AvgPool1d(CONV1D_MAX_SIZE)
        # or
        # x = torch.flatten(x, N)  # shape n_data, encoder_out_dim, N to shape n_data, encoder_out_dim * N

        conv_1d_input_dim = CONV1D_FEATURE_DIM
        self.fc1 = nn.Linear(conv_1d_input_dim, CONV1D_FC1_OUTPUT_DIM)
        self.fc2 = nn.Linear(CONV1D_FC1_OUTPUT_DIM, CONV1D_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(CONV1D_FC2_OUTPUT_DIM, num_classes)

    def forward(self, x):
        # assert x.shape is (BS, In_CHNL, ~80000) --> it is!
        # assert In_CHNL is 1 or 2 --> it is 1.
        # nn.Conv1D: (N, Cin, Lin) -> (N, Cout, Lout)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # AdaptativeMaxPooling
        # Max_pool expected input is (N, Cout, Lout)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)

        # Classification
        # Expects shape (N_data, fc1_input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNClassifier(nn.Module):
    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes):
        super(RNNClassifier, self).__init__()
        # first encoder
        # neural audio embeddings
        # captures local representations through convolutions
        # note: x.shape is (bs, 1, ~80000)
        n_layers = int(math.log2(RNN1D_DOWNSAMPLER_OUT_CHANNELS))
        self.conv_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            # assumes input is monoaural i.e. shape is (bs, 1, len)
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=2 ** layer_idx,
                    out_channels=2 ** (layer_idx + 1),
                    kernel_size=RNN1D_DOWNSAMPLER_KERNEL_SIZE,
                    stride=RNN1D_DOWNSAMPLER_STRIDE,
                    dilation=RNN1D_DOWNSAMPLER_DILATION
                )
            )
        self.rnn = nn.LSTM(
            input_size=RNN1D_DOWNSAMPLER_OUT_CHANNELS,
            hidden_size=RNN1D_HIDDEN_SIZE,
            num_layers=RNN1D_LSTM_LAYERS,
            dropout=RNN1D_DROPOUT_PROB,
            bidirectional=RNN1D_BIDIRECTIONAL
        )

        self.max_pool = nn.AdaptiveAvgPool1d(RNN1D_MAX_SIZE)
        self.avg_pool = nn.AvgPool1d(RNN1D_MAX_SIZE)

        self.fc1 = nn.Linear(RNN1D_FC1_INPUT_SIZE, RNN1D_FC1_OUTPUT_SIZE)
        self.fc2 = nn.Linear(RNN1D_FC1_OUTPUT_SIZE, RNN1D_FC2_OUTPUT_SIZE)
        self.fc3 = nn.Linear(RNN1D_FC2_OUTPUT_SIZE, num_classes)

    def forward(self, x):
        # assert x.shape is (BS, In_CHNL, ~80000) --> it is!
        # assert In_CHNL is 1 or 2 --> it is 1.
        # nn.Conv1D: (N, Cin, Lin) -> (N, Cout, Lout)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # question for the reader:
        # Why PyTorch have different input shape for CNNs (N, Cin, Lin) compared to RNNs (Lin, N, Cin)
        x = x.transpose(0, 2).transpose(1, 2)  # (N, Cout, Lout) -> (Lout, Cout, N) -> (Lout, N, Cout)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # shape n_sequence, n_data, lstm_hidden_size (dropped _ is (h_n, c_n))

        x = x.transpose(1, 0)  # (Lout, N, Cout) -> (N, Lout, Cout)
        x = x.transpose(1, 2)  # (N, Lout, Cout) -> (N, Cout, Lout)

        # AdaptativeMaxPooling
        # Max_pool expected input is (N, Cout, Lout)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WaveNetLSTMClassifier(nn.Module):

    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes):
        super(WaveNetLSTMClassifier, self).__init__()
        # first encoder
        # neural audio embeddings
        # captures local representations through convolutions
        self.wavenet = WaveNetModel(
            WNLSTM_WAVENET_LAYERS,
            WNLSTM_WAVENET_BLOCKS,
            WAVENET_DILATION_CHANNELS,
            WAVENET_RESIDUAL_CHANNELS,
            WAVENET_SKIP_CHANNELS,
            WAVENET_END_CHANNELS,
            WAVENET_CLASSES,
            WAVENET_OUTPUT_LENGTH,
            WAVENET_KERNEL_SIZE
        )
        # Conv1d to reduce sequence length from 180k to 2k
        stride = 64
        self.conv1d_1 = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=256,
            kernel_size=4,
            stride=stride,
            dilation=16
        )
        self.conv1d_list = [self.conv1d_1, ]
        self.enc_lstm = nn.LSTM(
            256,
            LSTM_HIDDEN_SIZE,
            LSTM_NUM_LAYERS,
            bidirectional=LSTM_BIDIRECTIONALITY,
            dropout=LSTM_DROPOUT_PROB)

        self.max_pool = nn.AdaptiveAvgPool1d(WNLSTM_MAX_SIZE)
        self.avg_pool = nn.AvgPool1d(WNLSTM_MAX_SIZE)

        self.fc1 = nn.Linear(
            LSTM_HIDDEN_SIZE * 2 if LSTM_BIDIRECTIONALITY else LSTM_HIDDEN_SIZE,
            LSTM_FC1_OUTPUT_DIM
        )
        self.fc2 = nn.Linear(LSTM_FC1_OUTPUT_DIM, LSTM_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(LSTM_FC2_OUTPUT_DIM, num_classes)

    def forward(self, x):
        x = self.wavenet.forward(x)
        for conv1d_layer in self.conv1d_list:
            x = conv1d_layer(x)
        # x.shape is n_data, n_channels, n_sequence
        # rnn expected input is n_sequence, n_data, wavenet_channels
        x = x.transpose(0, 2).transpose(1, 2)
        # print('info: feeding lstm...')
        self.enc_lstm.flatten_parameters()
        x, _ = self.enc_lstm(x)  # shape n_sequence, n_data, lstm_hidden_size * 2
        x = x.transpose(0, 1)  # (Lout, N, Cout, ) -> (N, Lout, Cout, )
        x = x.transpose(1, 2)  # (N, Lout, Cout, ) -> (N, Cout, Lout, )
        # AdaptativeMaxPooling
        # Max_pool expected input is (N, Cout, Lout)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)
        # x final shape is n_data, lstm_hidden_size * 2
        # simple classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WaveNetTransformerClassifier(nn.Module):

    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes):
        super(WaveNetTransformerClassifier, self).__init__()
        # first encoder
        # neural audio embeddings
        # captures local representations through convolutions
        self.wavenet = WaveNetModel(
            WNTF_WAVENET_LAYERS,
            WNTF_WAVENET_BLOCKS,
            WAVENET_DILATION_CHANNELS,
            WAVENET_RESIDUAL_CHANNELS,
            WAVENET_SKIP_CHANNELS,
            WAVENET_END_CHANNELS,
            WAVENET_CLASSES,
            WAVENET_OUTPUT_LENGTH,
            WAVENET_KERNEL_SIZE)
        # Conv1d to reduce sequence length from 180k to 2k
        self.conv1d_1 = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=256,
            kernel_size=4,
            stride=64,
            dilation=16
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=12,
            stride=12
        )
        self.conv1d_list = [self.conv1d_1, self.conv1d_2]
        max_seq_len = int(
            math.ceil(WAVEFORM_RANDOM_CROP_SEQUENCE_LENGTH / 64 / 12)
        )
        self.positional_encoder = PositionalEncoding(WNTF_TRANSFORMER_D_MODEL, dropout=0.1, max_len=max_seq_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=WNTF_TRANSFORMER_D_MODEL,
                nhead=WNTF_TRANSFORMER_N_HEAD,
                dim_feedforward=WNTF_TRANSFORMER_DIM_FEEDFORWARD,
            )
            ,
            num_layers=WNTF_TRANSFORMER_N_LAYERS
        )

        self.max_pool = nn.AdaptiveAvgPool1d(WNTF_MAX_SIZE)
        self.avg_pool = nn.AvgPool1d(WNTF_MAX_SIZE)

        self.fc1 = nn.Linear(WNTF_TRANSFORMER_D_MODEL, WNTF_FC1_OUTPUT_DIM)
        self.fc2 = nn.Linear(WNTF_FC1_OUTPUT_DIM, WNTF_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(WNTF_FC2_OUTPUT_DIM, num_classes)

    def forward(self, x):
        x = self.wavenet.forward(x)
        for conv1d_layer in self.conv1d_list:
            x = conv1d_layer(x)
        # x.shape for convs is n_data, Cout, Lout

        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = x.transpose(1, 2)  # (N, Cout, Lout, ) to (N, Lout, Cout)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)  # shape  n_data, n_sequence, d_model

        # Max_pool expected input is (N, Cout, Lout)
        x = x.transpose(1, 2)  # (N, Lout, Cout, ) to (N, Cout, Lout)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WaveNetClassifier(nn.Module):

    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes):
        super(WaveNetClassifier, self).__init__()
        # first encoder
        # neural audio embeddings
        # captures local representations through convolutions
        self.wavenet = WaveNetModel(
            WAVENET_LAYERS,
            WAVENET_BLOCKS,
            WAVENET_DILATION_CHANNELS,
            WAVENET_RESIDUAL_CHANNELS,
            WAVENET_SKIP_CHANNELS,
            WAVENET_END_CHANNELS,
            WAVENET_CLASSES,
            WAVENET_OUTPUT_LENGTH,
            WAVENET_KERNEL_SIZE)
        # for now output length is fixed to 159968

        self.max_pool = nn.AdaptiveAvgPool1d(WN_MAX_SIZE)
        self.avg_pool = nn.AvgPool1d(WN_MAX_SIZE)

        self.fc1 = nn.Linear(self.wavenet.end_channels,
                             WAVENET_FC1_OUTPUT_DIM)  # 6*6 from image dimension
        self.fc2 = nn.Linear(WAVENET_FC1_OUTPUT_DIM, WAVENET_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(WAVENET_FC2_OUTPUT_DIM, num_classes)
        self.wavenet_pooling = 'amax'

    def forward(self, x):
        # Encoder
        # shape is (batch_size, channel, sequence_number)
        x = self.wavenet.forward(x)  # (N, Cout, Lout)
        # AvgPooling all the sequences into one Audio Embeding
        if self.wavenet_pooling == 'mean':
            x = torch.mean(x, dim=2)
        elif self.wavenet_pooling == 'max':
            x, _ = torch.max(x, dim=2)
        elif self.wavenet_pooling == 'amax':
            x = self.max_pool(x)
            x = self.avg_pool(x)
            x = x.squeeze(2)
        else:
            raise NotImplementedError()
        # shape is (batch_size, wavenet_out_channel)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


features_data_path = FEATURES_DATA_PATH
