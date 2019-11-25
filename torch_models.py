import argparse
import math
import os
import pathlib
import typing
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from math import ceil, floor

import numpy as np
import pytorch_lightning as ptl
import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import MODELS_DATA_PATH, S1DCONV_EPOCHS, S1DCONV_BATCH_SIZE, WAVENET_EPOCHS, WAVENET_BATCH_SIZE, \
    WAVENET_LAYERS, WAVENET_BLOCKS, WAVENET_DILATION_CHANNELS, WAVENET_RESIDUAL_CHANNELS, WAVENET_SKIP_CHANNELS, \
    WAVENET_END_CHANNELS, WAVENET_CLASSES, WAVENET_OUTPUT_LENGTH, WAVENET_KERNEL_SIZE, WAVENET_POOLING_KERNEL_SIZE, \
    WAVENET_POOLING_STRIDE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT_PROB, WAVEFORM_MAX_SEQUENCE_LENGTH, \
    TRANSFORMER_D_MODEL, TRANSFORMER_N_HEAD, TRANSFORMER_N_LAYERS, NUM_WORKERS, WAVEFORM_NUM_CHANNELS, makedirs, \
    FEATURES_DATA_PATH
from features import SingingVoiceSeparationOpenUnmixFeatureExtractor
from loaders import WaveformDataset
from models import ClassificationModel
from util.gmm_torch.gmm import GaussianMixture
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


class WaveNetClassifier(TorchClassificationModel):
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
        self.last_pooling = nn.AvgPool1d(kernel_size=10, stride=5)

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
        x = self.soft_max(x)
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
            WAVENET_LAYERS,
            WAVENET_BLOCKS,
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
        # output_length = floor((input_length - stride)/kernel_size + 1)
        conv_downsampler_stride = 3
        self.conv_downsampler_1 = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=384,
            kernel_size=4,
            stride=conv_downsampler_stride,
            dilation=2
        )
        self.conv_downsampler_2 = nn.Conv1d(
            in_channels=384,
            out_channels=448,
            kernel_size=4,
            stride=conv_downsampler_stride,
            dilation=2
        )
        self.conv_downsampler_3 = nn.Conv1d(
            in_channels=448,
            out_channels=d_model,
            kernel_size=4,
            stride=conv_downsampler_stride,
            dilation=2
        )

        self.positional_encoder = PositionalEncoder(
            d_model,
            max_seq_len=math.floor(max_raw_sequnece / (conv_downsampler_stride * 3))
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # self.to(self.device)
        # self.wavenet.to(self.device)
        # self.positional_encoder.to(self.device)
        # encoder_layer.to(self.device)
        # self.transformer_encoder.to(self.device)

    def forward(self, x):
        # print('info: feeding wavenet...')
        x = self.wavenet.forward(x)
        # reduce sequence_length / 10 three times == 16Khz to 10Hz; increase the number of channels
        x = self.conv_downsampler_1(x)
        x = self.conv_downsampler_2(x)
        x = self.conv_downsampler_3(x)
        # x.shape for convs is n_data, n_channels, n_sequence
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = x.transpose(1, 2)
        # print('info: feeding positional encoder...')
        x = self.positional_encoder(x)
        # print('info: feeding transformer...')
        x = self.transformer_encoder(x)  # shape  n_data, n_sequence, d_model
        x = x[:, -1, :]  # pick the last vector from the output as the sentence embedding
        # x final shape is n_data, lstm_hidden_size * 2
        # print('info: feeding fully-connected...')
        # simple classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class L_WavenetTransformerClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_WavenetTransformerClassifier, self).__init__()
        self.hparams = hparams
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = WaveNetTransformerClassifier(num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [self.optimizer]

    # def __dataloader(self, train):
    #     # init data generators
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.5,), (1.0,))])
    #     dataset = MNIST(root=self.hparams.data_root, train=train,
    #                     transform=transform, download=True)
    #
    #     # when using multi-node (ddp) we need to add the  datasampler
    #     train_sampler = None
    #     batch_size = self.hparams.batch_size
    #
    #     if self.use_ddp:
    #         train_sampler = DistributedSampler(dataset)
    #
    #     should_shuffle = train_sampler is None
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=0
    #     )
    #
    #     return loader

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(self.train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(self.eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(self.test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=WAVENET_BATCH_SIZE, type=int)
        return parser

class GMMClassifier(nn.Module):

    def __call__(self, *input, **kwargs) -> typing.Any:
        """
        Hack to fix '(input: (Any, ...), kwargs: dict) -> Any' warning in PyCharm auto-complete.
        :param input:
        :param kwargs:
        :return:
        """
        return super().__call__(*input, **kwargs)

    def __init__(self, num_classes):
        super(GMMClassifier, self).__init__()
        # first encoder
        # neural audio embeddings
        # captures local representations through convolutions
        n_features = None  # todo: fill this

        self.gmm_list = []
        for _ in num_classes:
            # one gmm per singer as stated in Tsai; Fujihara; Mesaros et. al works on SID
            self.gmm_list.append(GaussianMixture(n_components=64, n_features=n_features))

    def forward_score(self, x):
        """
        :param x: MFCC of a track with shape (samples, frames, coefficients, )
        :return: The Log Likelihood for each track and frame tested on every GMM (one per singer / class) as:
            log likelihood = log p(X_t / P_i)]

            where t is time frame and
                i is the singer GMM

            with shape: (sample, frame, gmm_prediction)
        """
        n_samples = x.size(0)
        # asume that all the samples has equal frame number
        n_frames = x.size(1)
        n_features = x.size(2)
        score_tensors_per_gmm = []
        # print('info: feeding gmms...')
        for gmm in self.gmm_list:
            # predict each frame for each sampple
            # optimization: flatten (samples, frames, features) to (samples*frames, features)
            x = x.reshape(-1, n_features)
            log_prob = gmm.score(x)  # output shape is (samples*frames, )
            log_prob = torch.unsqueeze(log_prob, dim=1)  # output shape is (samples*frames, 1) for stacking
            score_tensors_per_gmm.append(
                log_prob
            )
        score_tensors_per_gmm = torch.stack(score_tensors_per_gmm,
                                            dim=1)  # output tensor shape is ((samples*frames, gmm)
        y = score_tensors_per_gmm.reshape(
            n_samples,
            n_frames,
            len(self.gmm_list)
        )  # reshape to (sample, frame, gmm)
        return y

    def forward(self, x):
        """
        :param x: MFCC of a track with shape (samples, frames, coefficients, )
        :return: The prediction for each track calculated as:
            Singer_id = arg max_i [1/T sum^T_t=1 [log p(X_t / P_i)]]

            where t is time frame and
                i is the singer GMM

            with shape: (n_samples, )
        """
        x = self.forward_score(x)  # shape is (sample, frame, gmm)
        x = x.sum(dim=1)  # shape is (sample, gmm)
        x = x.argmax(dim=1)  # shape is (sample, )
        # sum the scores over the
        return x


class L_GMMClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_GMMClassifier, self).__init__()
        self.hparams = hparams
        # self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = GMMClassifier(num_classes)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch['x'], batch['y']

        for gmm_idx, gmm in enumerate(self.model.gmm_list):
            gmm.fit()


        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [self.optimizer]

    # def __dataloader(self, train):
    #     # init data generators
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.5,), (1.0,))])
    #     dataset = MNIST(root=self.hparams.data_root, train=train,
    #                     transform=transform, download=True)
    #
    #     # when using multi-node (ddp) we need to add the  datasampler
    #     train_sampler = None
    #     batch_size = self.hparams.batch_size
    #
    #     if self.use_ddp:
    #         train_sampler = DistributedSampler(dataset)
    #
    #     should_shuffle = train_sampler is None
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=0
    #     )
    #
    #     return loader

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(self.train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(self.eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(self.test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=WAVENET_BATCH_SIZE, type=int)
        return parser


features_data_path = FEATURES_DATA_PATH
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model from features from a data folder',
        add_help=False
    )

    parser.add_argument(
        '--model',
        help='name of the model to be trained (options: ResNetV2, leglaive)',
        default='waveNetTransformer'
    )

    parser.add_argument('--features_path', help='Path to features folder',
                        default=features_data_path)

    parser.add_argument(
        '--experiment',
        default='unnamed_experiment',
        help='Name of the experiment. affects checkpoint names')

    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='how many gpus'
    )

    parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )

    args = parser.parse_args()
    model = args.model
    features_path = pathlib.Path(args.features_path)
    experiment_name = args.experiment

    print('info: feature_path is {}'.format(features_path))
    print('info: experiment_name is {}'.format(experiment_name))

    if model == 'Simple1dConvNet':
        train_dataset, test_dataset, eval_dataset = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(eval_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (S1DCONV_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_MAX_SEQUENCE_LENGTH)
        model = Simple1dConvNet(
            'faith_tull_binary2',
            num_classes=2,
            input_shape=input_shape
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, eval_dataset)
        model.evaluate(test_dataset)
    elif model == 'waveNet':
        train_dataset, test_dataset, eval_dataset = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(eval_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (S1DCONV_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_MAX_SEQUENCE_LENGTH)
        model = WaveNetClassifier(
            'faith_tull_binary2',
            num_classes=2,
            input_shape=input_shape
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, eval_dataset)
        model.evaluate(test_dataset)
    elif model == 'waveNetLstm':
        train_dataset, test_dataset, eval_dataset, number_of_classes = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (WAVENET_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_MAX_SEQUENCE_LENGTH)
        model = WaveNetBiLSTMClassifier(
            experiment_name,
            num_classes=number_of_classes,
            input_shape=input_shape,
            device_name=device_name
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, eval_dataset)
        model.evaluate(test_dataset)
    elif model == 'waveNetTransformer':
        root_dir = os.path.dirname(os.path.realpath(__file__))
        train_dataset, test_dataset, eval_dataset, number_of_classes = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        parser = WavenetTramsformerClassifier.add_model_specific_args(parser, root_dir)
        hyperparams = parser.parse_args()

        model = WavenetTramsformerClassifier(
            hyperparams,
            number_of_classes,
            train_dataset,
            eval_dataset,
            test_dataset
        )
        makedirs(MODELS_DATA_PATH / experiment_name)
        logger = ptl.logging.TestTubeLogger(
            save_dir=MODELS_DATA_PATH / experiment_name,
            version=1  # An existing version with a saved checkpoint
        )
        trainer = ptl.Trainer(
            gpus=hyperparams.gpus,
            distributed_backend=hyperparams.distributed_backend,
            logger=logger,
            default_save_path=MODELS_DATA_PATH / experiment_name,
            early_stop_callback=None
        )
        trainer.fit(model)