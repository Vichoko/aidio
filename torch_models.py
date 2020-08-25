import math
import typing
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from sklearn import mixture
from sklearn.exceptions import NotFittedError
from torch import nn as nn
from torch.nn import functional as F

from config import WAVENET_LAYERS, WAVENET_BLOCKS, WAVENET_DILATION_CHANNELS, WAVENET_RESIDUAL_CHANNELS, \
    WAVENET_SKIP_CHANNELS, \
    WAVENET_END_CHANNELS, WAVENET_CLASSES, WAVENET_OUTPUT_LENGTH, WAVENET_KERNEL_SIZE, WAVENET_POOLING_KERNEL_SIZE, \
    WAVENET_POOLING_STRIDE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT_PROB, WAVEFORM_MAX_SEQUENCE_LENGTH, \
    WNTF_TRANSFORMER_D_MODEL, WNTF_TRANSFORMER_N_HEAD, WNTF_TRANSFORMER_N_LAYERS, FEATURES_DATA_PATH, \
    GMM_COMPONENT_NUMBER, \
    GMM_FIT_FRAME_LIMIT, WNTF_WAVENET_LAYERS, WNTF_WAVENET_BLOCKS, \
    WNLSTM_WAVENET_LAYERS, WNLSTM_WAVENET_BLOCKS, CPU_NUM_WORKERS, CONV1D_FEATURE_DIM, RNN1D_DOWNSAMPLER_OUT_CHANNELS, \
    RNN1D_DROPOUT_PROB, RNN1D_HIDDEN_SIZE, RNN1D_LSTM_LAYERS, RNN1D_BIDIRECTIONAL, RNN1D_DOWNSAMPLER_STRIDE, \
    RNN1D_DOWNSAMPLER_KERNEL_SIZE, RNN1D_DOWNSAMPLER_DILATION, CONV1D_KERNEL_SIZE, CONV1D_STRIDE, CONV1D_DILATION, \
    LSTM_FC1_OUTPUT_DIM, LSTM_FC2_OUTPUT_DIM, CONV1D_FC1_OUTPUT_DIM, CONV1D_FC2_OUTPUT_DIM, WNTF_FC1_OUTPUT_DIM, \
    WNTF_FC2_OUTPUT_DIM, WNTF_TRANSFORMER_DIM_FEEDFORWARD, LSTM_BIDIRECTIONALITY
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


class PositionalEncoder(torch.nn.Module):
    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        # pe = torch.zeros(max_seq_len, d_model)
        pe = None
        for i in range(0, d_model, 2):
            pair = torch.sin(
                torch.Tensor([pos / (10000 ** ((2 * i) / d_model)) for pos in range(max_seq_len)])).reshape(-1, 1)
            even = torch.cos(
                torch.Tensor([pos / (10000 ** ((2 * (i + 1)) / d_model)) for pos in range(max_seq_len)])).reshape(-1, 1)
            if pe is None:
                pe = torch.cat([pair, even], dim=1)
            else:
                pe = torch.cat([pe, pair, even], dim=1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
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
        # reduce sample resolution from 160k to 32k
        # output_length = floor(
        #       (input_length - (kernel_size-1)) / stride + 1
        #   )
        self.conv_dimension_reshaper = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=WNTF_TRANSFORMER_D_MODEL,
            kernel_size=WAVENET_POOLING_KERNEL_SIZE,
            stride=WAVENET_POOLING_STRIDE
        )
        max_seq_len = int(math.floor(
            (WAVEFORM_MAX_SEQUENCE_LENGTH - (WAVENET_POOLING_KERNEL_SIZE - 1)) / WAVENET_POOLING_STRIDE + 1))
        self.positional_encoder = PositionalEncoder(
            WNTF_TRANSFORMER_D_MODEL,
            max_seq_len=max_seq_len
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=WNTF_TRANSFORMER_D_MODEL,
                nhead=WNTF_TRANSFORMER_N_HEAD,
                dim_feedforward=WNTF_TRANSFORMER_DIM_FEEDFORWARD,
            )
            ,
            num_layers=WNTF_TRANSFORMER_N_LAYERS
        )
        self.fc1 = nn.Linear(WNTF_TRANSFORMER_D_MODEL, WNTF_FC1_OUTPUT_DIM)
        self.fc2 = nn.Linear(WNTF_FC1_OUTPUT_DIM, WNTF_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(WNTF_FC2_OUTPUT_DIM, num_classes)

    def forward(self, x):
        x = self.wavenet.forward(x)
        x = self.conv_dimension_reshaper(x)
        # x.shape for convs is n_data, n_channels, n_sequence
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = x.transpose(1, 2)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)  # shape  n_data, n_sequence, d_model
        # x = x[:, -1, :]  # pick the last vector from the output as the sentence embedding
        x, _ = x.max(1)  # max pooling over the sequence dim; drop sequence axis
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        self.self_attention_pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=CONV1D_FEATURE_DIM,
                nhead=1,
            ),
            num_layers=1
        )  # take the last vector of the attention to the FC
        self.fc1 = nn.Linear(CONV1D_FEATURE_DIM, CONV1D_FC1_OUTPUT_DIM)
        self.fc2 = nn.Linear(CONV1D_FC1_OUTPUT_DIM, CONV1D_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(CONV1D_FC2_OUTPUT_DIM, num_classes)

    def forward(self, x):
        # assert x.shape is (BS, In_CHNL, ~80000) --> it is!
        # assert In_CHNL is 1 or 2 --> it is 1.
        # nn.Conv1D: (N, Cin, Lin) -> (N, Cout, Lout)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.transpose(1, 2)  # (N, Cout, Lout) -> (N, Lout, Cout)
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = self.self_attention_pooling(x)
        x = x[:, -1, :]
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
        self.self_attention_pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=RNN1D_HIDDEN_SIZE * 2 if RNN1D_BIDIRECTIONAL else RNN1D_HIDDEN_SIZE,
                nhead=1,
            ),
            num_layers=1
        )  # take the last vector of the attention to the FC
        self.fc1 = nn.Linear(RNN1D_HIDDEN_SIZE * 2, 256) if RNN1D_BIDIRECTIONAL else nn.Linear(
            RNN1D_HIDDEN_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # assert x.shape is (BS, In_CHNL, ~80000) --> it is!
        # assert In_CHNL is 1 or 2 --> it is 1.
        # nn.Conv1D: (N, Cin, Lin) -> (N, Cout, Lout)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # question for the reader:
        # Why PyTorch have different input shape for CNNs (N, Cin, Lin) compared to RNNs (Lin, N, Cin)
        x = x.transpose(0, 2).transpose(1, 2)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # shape n_sequence, n_data, lstm_hidden_size (dropped _ is (h_n, c_n))
        x = x.transpose(0, 1)
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = self.self_attention_pooling(x)
        x = x[:, -1, :]
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
            WAVENET_KERNEL_SIZE)
        # reduce sample resolution from 160k to 32k
        # output_length = floor((input_length - (kernel_size - 1))/stride + 1)
        self.conv_dimension_reshaper = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=WAVENET_END_CHANNELS,
            kernel_size=WAVENET_POOLING_KERNEL_SIZE,
            stride=WAVENET_POOLING_STRIDE
        )
        self.enc_lstm = nn.LSTM(
            self.wavenet.end_channels,
            LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
            bidirectional=LSTM_BIDIRECTIONALITY,
            dropout=LSTM_DROPOUT_PROB)

        self.self_attention_pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=LSTM_HIDDEN_SIZE * 2 if LSTM_BIDIRECTIONALITY else LSTM_HIDDEN_SIZE,
                nhead=1,
                dim_feedforward=WNTF_TRANSFORMER_DIM_FEEDFORWARD,
            )
            ,
            num_layers=1
        )  # take the last vector of the attention to the FC

        self.fc1 = nn.Linear(
            LSTM_HIDDEN_SIZE * 2 if LSTM_BIDIRECTIONALITY else LSTM_HIDDEN_SIZE,
            LSTM_FC1_OUTPUT_DIM
        )
        self.fc2 = nn.Linear(LSTM_FC1_OUTPUT_DIM, LSTM_FC2_OUTPUT_DIM)
        self.fc3 = nn.Linear(LSTM_FC2_OUTPUT_DIM, num_classes)

    def forward(self, x):
        # print('info: feeding wavenet...')
        x = self.wavenet.forward(x)
        # reduce sequence_length / 5
        x = self.conv_dimension_reshaper(x)
        # x.shape is n_data, n_channels, n_sequence
        # rnn expected input is n_sequence, n_data, wavenet_channels
        x = x.transpose(0, 2).transpose(1, 2)
        # print('info: feeding lstm...')
        self.enc_lstm.flatten_parameters()
        x, _ = self.enc_lstm(x)  # shape n_sequence, n_data, lstm_hidden_size * 2
        x = x.transpose(0, 1)
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = self.self_attention_pooling(x)
        x = x[:, -1, :]
        # pick the last vector from the output as the sentence embedding
        # x final shape is n_data, lstm_hidden_size * 2
        # print('info: feeding fully-connected...')
        # simple classifier
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
        self.fc1 = nn.Linear(self.wavenet.end_channels,
                             120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.wavenet_pooling = 'max'

    def forward(self, x):
        # Encoder
        # shape is (batch_size, channel, sequence_number)
        x = self.wavenet.forward(x)
        # AvgPooling all the sequences into one Audio Embeding
        if self.wavenet_pooling == 'mean':
            x = torch.mean(x, dim=2)
        elif self.wavenet_pooling == 'max':
            x, _ = torch.max(x, dim=2)
        else:
            raise NotImplementedError()
        # shape is (batch_size, wavenet_out_channel)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


features_data_path = FEATURES_DATA_PATH
