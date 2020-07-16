import math
import typing
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from math import ceil, floor

import numpy as np
import torch
import tqdm
from sklearn import mixture
from sklearn.exceptions import NotFittedError
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import MODELS_DATA_PATH, S1DCONV_EPOCHS, S1DCONV_BATCH_SIZE, WAVENET_EPOCHS, WAVENET_BATCH_SIZE, \
    WAVENET_LAYERS, WAVENET_BLOCKS, WAVENET_DILATION_CHANNELS, WAVENET_RESIDUAL_CHANNELS, WAVENET_SKIP_CHANNELS, \
    WAVENET_END_CHANNELS, WAVENET_CLASSES, WAVENET_OUTPUT_LENGTH, WAVENET_KERNEL_SIZE, WAVENET_POOLING_KERNEL_SIZE, \
    WAVENET_POOLING_STRIDE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT_PROB, WAVEFORM_MAX_SEQUENCE_LENGTH, \
    TRANSFORMER_D_MODEL, TRANSFORMER_N_HEAD, TRANSFORMER_N_LAYERS, FEATURES_DATA_PATH, GMM_COMPONENT_NUMBER, \
    GMM_FIT_FRAME_LIMIT, WNTF_WAVENET_LAYERS, WNTF_WAVENET_BLOCKS, \
    WNLSTM_WAVENET_LAYERS, WNLSTM_WAVENET_BLOCKS, CPU_NUM_WORKERS
from models import ClassificationModel
from util.wavenet.wavenet_model import WaveNetModel


class TorchClassificationModel(ClassificationModel, nn.Module):
    """
    Provides the train, evaluation, checkpoints behaviour to the architecture.
    """
    model_name = 'TorchClassificationModel_unspecified'

    def __init__(self, model_type, num_classes, input_shape, model_path,
                 epochs,
                 batch_size,
                 device_name='cuda:0',
                 **kwargs):
        ClassificationModel.__init__(self,
                                     self.model_name,
                                     model_type,
                                     num_classes,
                                     input_shape,
                                     model_path,
                                     epochs,
                                     batch_size,
                                     **kwargs)
        nn.Module.__init__(self)
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        print('info: using {} for this model'.format(self.device))
        self.best_loss = float('inf')

    def post_epoch(self, epoch, **kwargs):
        """
        Called in between-epochs.
        Evaluate, Save checkpoint and check early stop by default.
        :return:
        """
        print("metric: finished epoch {}. Starting evaluation...".format(epoch))
        losses = kwargs['losses']
        train_mean_loss = np.mean(losses)
        print("metric: train mean loss: {}".format(train_mean_loss))
        val_dataset = kwargs['val_dataset']
        name = kwargs['name']
        val_loss = self.evaluate(val_dataset, name)
        self.train()
        self.save_checkpoint(epoch, val_loss)
        self.early_stop(epoch, val_loss)

    def early_stop(self, epoch, val_loss):
        """
        If val loss reach a minimum value, it stops the training
        to avoid overfitting
        :param epoch: Integer, number of current epoch
        :param val_loss: Float, loss on validation set
        :return:
        """

        # GL criteria
        gl = 100 * (val_loss / self.best_loss - 1)
        print('debug: early stopping gl = {}'.format(gl))

    def save_checkpoint(self, epoch, current_loss, save_best_only=True):
        """
        Save checkpoint of the model,
        :param epoch: Index of epoch
        :param current_loss: current train loss
        :param save_best_only: Bool if True save only if best is True else save always
        :return:
        """
        filename = str(self.model_checkpoint_path).format(epoch=epoch)
        if save_best_only == True and current_loss > self.best_loss:
            return
        print('info: saving checkpoint {}'.format(filename))
        torch.save(self, filename)
        self.best_loss = current_loss

    def train_now(self, train_dataset, val_dataset):
        """
        Trains the model giving useful metrics between epochs.

        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """
        print('info: training classifier...')
        self.train()
        batches_per_epoch = len(train_dataset) / self.batch_size
        quarter_epoch_batches = int(batches_per_epoch / 4)
        import torch.optim as optim

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=4)

        epoch = self.initial_epoch
        while epoch < self.epochs:  # loop over the dataset multiple times
            epoch += 1
            running_loss = 0.0
            losses = []
            pb = tqdm.tqdm(dataloader, desc='training in batches', unit='batch', position=0, leave=True)
            for i_batch, sample_batched in enumerate(pb):
                # get the inputs; data is a list of [inputs, labels]
                x_i = sample_batched['x'].to(self.device)
                y_i = sample_batched['y'].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(x_i)
                loss = criterion(outputs, y_i)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                losses.append(loss.item())
                if i_batch % quarter_epoch_batches == quarter_epoch_batches - 1:  # print every 2000 mini-batches
                    print('metric: [%d, %5d] train loss: %.3f' %
                          (epoch, i_batch + 1, running_loss / quarter_epoch_batches))
                    running_loss = 0.0
            # post-epoch behaviour
            feed_dict = {'val_dataset': val_dataset,
                         'name': 'validation',
                         'losses': losses
                         }
            self.post_epoch(epoch, **feed_dict)
        print('info: finished training by batch count')

    def evaluate(self, dataset, name='test'):
        """
        Evaluate model parameters against input data.
        It logs multi-class classification performance metrics in runtime.
        :param x: Input feed tensor
        :param y: Expected abels tensor
        :param name: Evaluation title name for logs.
        :return: Total Test set Loss
        """
        print('info: evaluating classifier with {} set...'.format(name))
        self.eval()
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

            losses = []
            total = 0
            metrics = defaultdict(lambda: {'hit': 0, 'total': 0})
            pb = tqdm.tqdm(dataloader, desc='evaluating in batches', unit='batch', position=0, leave=True)
            for i_batch, sample_batched in enumerate(pb):

                # get the inputs; data is a list of [inputs, labels]
                x = sample_batched['x'].to(self.device)
                labels = sample_batched['y'].to(self.device)

                # forward + backward + optimize
                outputs = self(x)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                matches = (predicted == labels).squeeze()
                for idx in range(labels.size(0)):
                    label = labels[idx]
                    metrics[label.item()]['hit'] += matches[idx].item() if matches.dim() > 0 else matches.item()
                    metrics[label.item()]['total'] += 1

            mean_eval_loss = np.mean(losses)
            print('metric: mean {} loss: {}'.format(name, mean_eval_loss))
            for key in metrics.keys():
                print('metric: %s accuracy of %5s : %2d %%' % (name,
                                                               key, 100 * metrics[key]['hit'] / metrics[key]['total']))
            return mean_eval_loss

    def predict(self, x):
        return self.forward(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Simple1dConvNet(TorchClassificationModel):
    def reset_hyper_parameters(self, model_name, model_type, num_classes, input_shape, initial_epoch,
                               model_path=MODELS_DATA_PATH,
                               epochs=S1DCONV_EPOCHS,
                               batch_size=S1DCONV_BATCH_SIZE):
        """
        Override Reset model hyper-parameters as epochs and batch_size.
        The purpose of this override is to set the default values for each configuration varible.
        :param model_name:
        :param model_type:
        :param num_classes:
        :param input_shape:
        :param initial_epoch:
        :param model_path:
        :param epochs:
        :param batch_size:
        :return:
        """
        super().reset_hyper_parameters(model_name, model_type, num_classes, input_shape, initial_epoch, model_path,
                                       epochs, batch_size)

    model_name = 'simple_1d_conv_net'

    def __init__(self, model_type, num_classes, input_shape, model_path=MODELS_DATA_PATH,
                 epochs=S1DCONV_EPOCHS,
                 batch_size=S1DCONV_BATCH_SIZE,
                 **kwargs):
        TorchClassificationModel.__init__(self,
                                          model_type,
                                          num_classes,
                                          input_shape,
                                          model_path,
                                          epochs,
                                          batch_size,
                                          **kwargs)

        assert len(self.input_shape) == 3  # (#, N_Channels, L)
        input_channels = self.input_shape[1]
        self.conv_kernel_size = 9
        self.pool_kernel_size = 4
        self.pool_stride = 2

        # 1 input image channel, 6 output channels, 9 linear convolution
        # kernel
        self.conv1 = nn.Conv1d(input_channels, 6, self.conv_kernel_size)
        self.conv2 = nn.Conv1d(6, 16, self.conv_kernel_size)
        self.pool = nn.MaxPool1d(self.pool_kernel_size, self.pool_stride)

        # calculate output shape of the encoder
        output_shape_l = self.input_shape[2]

        # first block
        output_shape_l = ceil((output_shape_l - self.conv_kernel_size + 1) / 1)
        output_shape_l = ceil((output_shape_l - self.pool_kernel_size + 1) / self.pool_stride)
        # second block
        output_shape_l = ceil((output_shape_l - self.conv_kernel_size + 1) / 1)
        output_shape_l = ceil((output_shape_l - self.pool_kernel_size + 1) / self.pool_stride)

        # classificator
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * output_shape_l, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

        # auxiliary state variables
        self.best_loss = float('inf')
        self.early_stop_flag = False

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OldWaveNetClassifier(TorchClassificationModel):
    model_name = 'wavenet_classif'

    def reset_hyper_parameters(self, model_name, model_type, num_classes, input_shape, initial_epoch,
                               model_path=MODELS_DATA_PATH,
                               epochs=WAVENET_EPOCHS,
                               batch_size=WAVENET_BATCH_SIZE):
        """
        Override Reset model hyper-parameters as epochs and batch_size.
        The purpose of this override is to set the default values for each configuration varible.
        :param model_name:
        :param model_type:
        :param num_classes:
        :param input_shape:
        :param initial_epoch:
        :param model_path:
        :param epochs:
        :param batch_size:
        :return:
        """
        super().reset_hyper_parameters(model_name, model_type, num_classes, input_shape, initial_epoch, model_path,
                                       epochs, batch_size)

    def __init__(self, model_type, num_classes, input_shape, model_path=MODELS_DATA_PATH,
                 epochs=WAVENET_EPOCHS,
                 batch_size=WAVENET_BATCH_SIZE,
                 **kwargs):
        TorchClassificationModel.__init__(self,
                                          model_type,
                                          num_classes,
                                          input_shape,
                                          model_path,
                                          epochs,
                                          batch_size,
                                          **kwargs)
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

        # reduce dim from 160k to 32k
        pooling_kz = 10
        pooling_stride = 5
        self.last_pooling = nn.AvgPool1d(kernel_size=pooling_kz, stride=pooling_stride)

        # for now output length is fixed to 159968

        self.fc1 = nn.Linear(self.wavenet.end_channels * floor((159968 - pooling_kz) / pooling_stride + 1),
                             120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.wavenet.forward(x)

        # reduce samples
        x = self.last_pooling(x)

        # simple classifier
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WaveNetBiLSTMClassifier(TorchClassificationModel):
    model_name = 'wavenet_disan_classif'

    def reset_hyper_parameters(self, model_name, model_type, num_classes, input_shape, initial_epoch,
                               model_path=MODELS_DATA_PATH,
                               epochs=WAVENET_EPOCHS,
                               batch_size=WAVENET_BATCH_SIZE):
        """
        Override Reset model hyper-parameters as epochs and batch_size.
        The purpose of this override is to set the default values for each configuration varible.
        :param model_name:
        :param model_type:
        :param num_classes:
        :param input_shape:
        :param initial_epoch:
        :param model_path:
        :param epochs:
        :param batch_size:
        :return:
        """
        super().reset_hyper_parameters(model_name, model_type, num_classes, input_shape, initial_epoch, model_path,
                                       epochs, batch_size)

    def __init__(self, model_type, num_classes, input_shape, model_path=MODELS_DATA_PATH,
                 epochs=WAVENET_EPOCHS,
                 batch_size=WAVENET_BATCH_SIZE,
                 device_name='cuda:0',
                 **kwargs):
        TorchClassificationModel.__init__(self,
                                          model_type,
                                          num_classes,
                                          input_shape,
                                          model_path,
                                          epochs,
                                          batch_size,
                                          device_name=device_name,
                                          **kwargs)

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

        # reduce sample resolution from 160k to 32k
        # output_length = floor((input_length - stride)/kernel_size + 1)
        self.avg_pooling = nn.AvgPool1d(
            kernel_size=WAVENET_POOLING_KERNEL_SIZE,
            stride=WAVENET_POOLING_STRIDE
        )

        self.enc_lstm = nn.LSTM(
            self.wavenet.end_channels,
            LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
            bidirectional=True,
            dropout=LSTM_DROPOUT_PROB)

        # for now output length is fixed to 159968

        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE * 2, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

        self.soft_max = nn.Softmax(dim=1)

        self.to(self.device)
        self.wavenet.to(self.device)

    def forward(self, x):
        # print('info: feeding wavenet...')
        x = self.wavenet.forward(x)
        # reduce sequence_length / 5
        x = self.avg_pooling(x)
        # x.shape is n_data, n_channels, n_sequence
        # rnn expected input is n_sequence, n_data, wavenet_channels
        x = x.transpose(0, 2).transpose(1, 2)
        # print('info: feeding lstm...')
        self.enc_lstm.flatten_parameters()
        x, _ = self.enc_lstm(x)  # shape n_sequence, n_data, lstm_hidden_size * 2
        x, _ = x.max(0)  # max pooling over the sequence dim; drop sequence axis
        # x final shape is n_data, lstm_hidden_size * 2
        # print('info: feeding fully-connected...')
        # simple classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.soft_max(x)
        return x


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

        max_raw_sequnece = WAVEFORM_MAX_SEQUENCE_LENGTH
        d_model = TRANSFORMER_D_MODEL
        nhead = TRANSFORMER_N_HEAD
        num_layers = TRANSFORMER_N_LAYERS

        # reduce sample resolution from 160k to 32k
        # output_length = floor(
        #       (input_length - (kernel_size-1)) / stride + 1
        #   )

        self.conv_dimension_reshaper = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=TRANSFORMER_D_MODEL,
            kernel_size=WAVENET_POOLING_KERNEL_SIZE,
            stride=WAVENET_POOLING_STRIDE
        )
        max_seq_len = math.floor(
            (max_raw_sequnece - (WAVENET_POOLING_KERNEL_SIZE - 1)) / WAVENET_POOLING_STRIDE + 1)
        self.positional_encoder = PositionalEncoder(
            d_model,
            max_seq_len=max_seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # print('info: feeding wavenet...')
        # todo: check input and outpit shapes of wavenet are OK
        x = self.wavenet.forward(x)
        x = self.conv_dimension_reshaper(x)
        # print('info: x after reshape {}'.format(x.size()))
        # x.shape for convs is n_data, n_channels, n_sequence
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = x.transpose(1, 2)
        # print('info: feeding positional encoder...')
        # x = self.positional_encoder(x)
        # print('info: feeding transformer...')
        # todo: check input and output shapes of t_e are OK
        x = self.transformer_encoder(x)  # shape  n_data, n_sequence, d_model
        # x = x[:, -1, :]  # pick the last vector from the output as the sentence embedding
        x, _ = x.max(1)  # max pooling over the sequence dim; drop sequence axis
        # x = x.mean(1)  # max pooling over the sequence dim; drop sequence axis
        # print('info. x shape {}'.format(x.shape))
        # x final shape is n_data, lstm_hidden_size * 2
        # print('info: feeding fully-connected...')
        # simple classifier
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
        encoder_kernel_size = 64
        encoder_stride = 64
        self.c1 = nn.Conv1d(
            in_channels=2 ** 0,
            out_channels=2 ** 2,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        self.c2 = nn.Conv1d(
            in_channels=2 ** 2,
            out_channels=2 ** 3,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        self.c3 = nn.Conv1d(
            in_channels=2 ** 3,
            out_channels=2 ** 4,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        self.c4 = nn.Conv1d(
            in_channels=2 ** 4,
            out_channels=2 ** 5,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        self.c5 = nn.Conv1d(
            in_channels=2 ** 5,
            out_channels=2 ** 6,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        self.fc1 = nn.Linear(2 ** 6, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # assert x.shape is (BS, In_CHNL, Sequence_L) --> It is! (Sequence_L) is 80,000
        # assert In_CHNL is 1 or 2 -- it is 1
        # nn.Conv1D: (N, Cin, Lin) -> (N, Cout, Lout)
        print(x.shape)
        x = self.c1(x)
        print(x.shape)
        x = self.c2(x)
        print(x.shape)
        x = self.c3(x)
        print(x.shape)
        x = self.c4(x)
        print(x.shape)
        x = self.c5(x)
        print(x.shape)
        ## x.max(2): (N, Cout, Lout) -> (N, Cout)
        x, _ = x.max(2)  # max pooling over the sequence dim; drop sequence axis
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
            bidirectional=True,
            dropout=LSTM_DROPOUT_PROB)
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

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
        x, _ = x.max(0)  # max pooling over the sequence dim; drop sequence axis
        # x final shape is n_data, lstm_hidden_size * 2
        # print('info: feeding fully-connected...')
        # simple classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
