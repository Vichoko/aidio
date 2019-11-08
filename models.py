import argparse
import math
import os
import pathlib
from collections import defaultdict
from functools import reduce
from math import ceil, floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data.dataloader import DataLoader

from config import MODELS_DATA_PATH, RESNET_V2_BATCH_SIZE, RESNET_V2_EPOCHS, RESNET_V2_DEPTH, \
    RESNET_V2_VERSION, SIMPLECONV_BATCH_SIZE, SIMPLECONV_EPOCHS, makedirs, FEATURES_DATA_PATH, WAVEFORM_NUM_CHANNELS, \
    WAVEFORM_SEQUENCE_LENGTH, S1DCONV_BATCH_SIZE, S1DCONV_EPOCHS, WAVENET_LAYERS, WAVENET_BLOCKS, \
    WAVENET_DILATION_CHANNELS, WAVENET_RESIDUAL_CHANNELS, WAVENET_SKIP_CHANNELS, WAVENET_OUTPUT_LENGTH, \
    WAVENET_KERNEL_SIZE, WAVENET_END_CHANNELS, WAVENET_CLASSES, WAVENET_EPOCHS, WAVENET_BATCH_SIZE, LSTM_HIDDEN_SIZE, \
    LSTM_NUM_LAYERS, LSTM_DROPOUT_PROB, WAVENET_POOLING_KERNEL_SIZE, WAVENET_POOLING_STRIDE, NUM_WORKERS, \
    TRANSFORMER_N_HEAD, TRANSFORMER_D_MODEL, TRANSFORMER_N_LAYERS
from features import WindowedMelSpectralCoefficientsFeatureExtractor, SingingVoiceSeparationOpenUnmixFeatureExtractor
from loaders import ResnetDataManager, TorchVisionDataManager, WaveformDataset
from util.wavenet.wavenet_model import WaveNetModel


class ClassificationModel:

    def __init__(self, model_name, model_type, num_classes, input_shape, model_path, epochs,
                 batch_size,
                 initial_epoch=0,
                 **kwargs):
        """

        :param model_name: String; name of the model class
        :param model_type: String; name of the model instance
        :param num_classes: Number of classes to predict
        :param input_shape: Tuple with shape of the input data (#_element, *data_dims).
                    Ex. (#_element, height, width, channels), generally ()
        :param model_path: Path or string to model files
        :param epochs: Integer of data-rounds
        :param batch_size: Size of batch for training
        :param kwargs:
        """
        self.model_name = model_name
        self.model_type = model_type

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.epochs = epochs
        model_filename = '%s_model.{epoch:03d}.h5' % self.model_name
        self.model_path = model_path / self.model_type
        makedirs(self.model_path)
        self.model_checkpoint_path = self.model_path / model_filename
        self.batch_size = batch_size
        self.initial_epoch = initial_epoch

    def reset_hyper_parameters(self, model_name, model_type, num_classes, input_shape, initial_epoch,
                               model_path,
                               epochs,
                               batch_size):
        ClassificationModel.__init__(self, model_name, model_type, num_classes, input_shape, model_path, epochs,
                                     batch_size,
                                     initial_epoch)

    @property
    def checkpoint_files(self):
        """
        Check for existing checkpoint for this model
        :return:
        """
        files = [f for f in os.listdir(self.model_checkpoint_path.parent)
                 if os.path.isfile(self.model_checkpoint_path.parent / f)]
        chkp_name = self.model_checkpoint_path.name.split('.')[0]
        paths = []
        epochs = []
        for file in files:
            name, epoch, _ = file.split('.')
            if name == chkp_name:
                paths.append(self.model_checkpoint_path.parent / file)
                epochs.append(int(epoch))
        return paths, epochs

    def remove_checkpoint(self):
        [os.remove(path) for path in self.checkpoint_files[0]]

    def load_checkpoint(self):
        """
        Load model object from last checkpoint if exist.
        :return:
        """
        # check for existing checkpoints
        chkp_files, chkp_epoch = self.checkpoint_files
        assert len(chkp_files) == len(chkp_epoch)
        if len(chkp_files) != 0:
            # load from checkpoint
            max_idx = int(np.argmax(chkp_epoch))
            model = torch.load(str(chkp_files[max_idx]))
            print('info: loading model from checkpoint {}'.format(chkp_files[max_idx]))
            initial_epoch = int(chkp_epoch[max_idx])
            model.reset_hyper_parameters(
                model.model_name,
                model.model_type,
                model.num_classes,
                model.input_shape,
                initial_epoch
            )
            return model
        return self


class ResNetV2(ClassificationModel):

    def __init__(self, model_name, num_classes, input_shape, model_path=MODELS_DATA_PATH,
                 epochs=RESNET_V2_EPOCHS, batch_size=RESNET_V2_BATCH_SIZE,
                 **kwargs):
        """
        resnet
        """
        # ResNet custom variables
        self.depth = RESNET_V2_DEPTH
        # Model name, depth and version
        model_type = 'ResNet%dv%d' % (self.depth, RESNET_V2_VERSION)
        super().__init__(model_name, model_type, num_classes, input_shape, model_path, epochs, batch_size, **kwargs)

        # loss deduction
        if self.num_classes > 2:
            self.loss = 'categorical_crossentropy'
        elif self.num_classes == 2:
            self.loss = 'binary_crossentropy'
        else:
            raise Exception('num_classes cant be lesser than 2...')

        # things needed to fit and predict
        self.model, self.callbacks = self.compile_model(
            **kwargs
        )

    def compile_model(self, **kwargs):
        """
        Compile the model to be ready to be trained.
        :return: Model, Callbacks for model operations
        """
        import keras
        from keras.engine.saving import load_model
        from keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten
        from keras.regularizers import l2
        from keras.optimizers import Adam
        from keras.models import Model
        from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

        def lr_schedule(epoch):
            """Learning Rate Schedule

            Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
            Called automatically every epoch as part of callbacks during training.

            # Arguments
                epoch (int): The number of epochs

            # Returns
                lr (float32): learning rate
            """
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder

            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)

            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        def resnet_v2(input_shape, depth, num_classes):
            """ResNet Version 2 Model builder [b]

            Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
            bottleneck layer
            First shortcut connection per layer is 1 x 1 Conv2D.
            Second and onwards shortcut connection is identity.
            At the beginning of each stage, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filter maps is
            doubled. Within each stage, the layers have the same number filters and the
            same filter map sizes.
            Features maps sizes:
            conv1  : 32x32,  16
            stage 0: 32x32,  64
            stage 1: 16x16, 128
            stage 2:  8x8,  256

            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)

            # Returns
                model (Model): Keras model instance
            """
            if (depth - 2) % 9 != 0:
                raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
            # Start model definition.
            num_filters_in = 16
            num_res_blocks = int((depth - 2) / 9)

            inputs = Input(shape=input_shape)
            # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
            x = resnet_layer(inputs=inputs,
                             num_filters=num_filters_in,
                             conv_first=True)

            # Instantiate the stack of residual units
            for stage in range(3):
                for res_block in range(num_res_blocks):
                    activation = 'relu'
                    batch_normalization = True
                    strides = 1
                    if stage == 0:
                        num_filters_out = num_filters_in * 4
                        if res_block == 0:  # first layer and first stage
                            activation = None
                            batch_normalization = False
                    else:
                        num_filters_out = num_filters_in * 2
                        if res_block == 0:  # first layer but not first stage
                            strides = 2  # downsample

                    # bottleneck residual unit
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters_in,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation,
                                     batch_normalization=batch_normalization,
                                     conv_first=False)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters_in,
                                     conv_first=False)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     conv_first=False)
                    if res_block == 0:
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                         num_filters=num_filters_out,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                    x = keras.layers.add([x, y])

                num_filters_in = num_filters_out

            # Add classifier on top.
            # v2 has BN-ReLU before Pooling
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model

        print('info: initiating classifier...')
        print('info: looking for checkpoints...')
        print('info: compiling model {}...'.format(self.model_name))
        # Input image dimensions.
        input_shape = self.input_shape[1:]

        if input_shape[1] < 29 or input_shape[0] < 29:
            print("warning: when a side of the img is less than 29, this network can't compile. shape= {}".format(
                input_shape))
            print("warning: pad the input data to at least 29 x 29")
            raise Exception("input shape should be at least 29 x 29")

        chkp_files, chkp_epoch = self.checkpoint_files
        assert len(chkp_files) == len(chkp_epoch)
        if len(chkp_files) != 0:
            # load from checkpoint
            max_idx = int(np.argmax(chkp_epoch))
            model = load_model(str(chkp_files[max_idx]))
            self.initial_epoch = int(chkp_epoch[max_idx])
        else:
            model = resnet_v2(input_shape=input_shape, depth=self.depth, num_classes=self.num_classes)
            model.compile(loss=self.loss,
                          optimizer=Adam(lr=lr_schedule(0)),
                          metrics=['accuracy'])
        model.summary()
        print(self.model_type)
        # Prepare model model saving directory.
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=str(self.model_checkpoint_path),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        return model, callbacks

    def train_now(self, x_train, y_train, x_test, y_test):
        print('info: training classifier...')
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True,
                       callbacks=self.callbacks,
                       initial_epoch=self.initial_epoch)

    def evaluate(self, x_test, y_test):
        print('info: evaluating...')
        # Score trained model.
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def forward(self, x):
        return self.model.predict(x)


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
                    metrics[label.item()]['hit'] += matches[idx].item()
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


class Simple2dConvNet(ClassificationModel, nn.Module):
    model_name = 'simple_2d_conv_net'

    def __init__(self, model_name, model_type, num_classes, input_shape, model_path=MODELS_DATA_PATH,
                 epochs=SIMPLECONV_EPOCHS,
                 batch_size=SIMPLECONV_BATCH_SIZE,
                 **kwargs):
        ClassificationModel.__init__(self, model_name, model_type, num_classes, input_shape, model_path, epochs,
                                     batch_size,
                                     **kwargs)
        nn.Module.__init__(self)

        assert len(self.input_shape) == 4  # (#, Channels, H, W)
        input_channels = self.input_shape[1]
        self.conv_kernel_size = 3
        self.pool_kernel_size = 2
        self.pool_stride = 2

        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_channels, 6, self.conv_kernel_size)
        self.conv2 = nn.Conv2d(6, 16, self.conv_kernel_size)
        self.pool = nn.MaxPool2d(self.pool_kernel_size, self.pool_stride)

        # calculate output shape of the encoder
        output_shape_h = self.input_shape[2]
        output_shape_w = self.input_shape[3]

        # first block
        output_shape_h = ceil((output_shape_h - self.conv_kernel_size + 1) / 1)
        output_shape_w = ceil((output_shape_w - self.conv_kernel_size + 1) / 1)
        output_shape_h = ceil((output_shape_h - self.pool_kernel_size + 1) / self.pool_stride)
        output_shape_w = ceil((output_shape_w - self.pool_kernel_size + 1) / self.pool_stride)
        # second block
        output_shape_h = ceil((output_shape_h - self.conv_kernel_size + 1) / 1)
        output_shape_w = ceil((output_shape_w - self.conv_kernel_size + 1) / 1)
        output_shape_h = ceil((output_shape_h - self.pool_kernel_size + 1) / self.pool_stride)
        output_shape_w = ceil((output_shape_w - self.pool_kernel_size + 1) / self.pool_stride)

        # classificator
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * output_shape_h * output_shape_w, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

        # auxiliary state variables
        self.early_stop_flag = False
        self.best_loss = float('inf')

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

    def post_epoch(self, epoch, **kwargs):
        """
        Called in between-epochs.
        Evaluate, Save checkpoint and check early stop by default.
        :return:
        """
        print("metric: finished epoch {}. Starting evaluation...".format(epoch))
        losses = kwargs['losses']
        batch_train_loss = reduce(lambda x, y: x + y, losses)
        print("metric: epoch total train loss: {}".format(batch_train_loss))
        x_val = kwargs['x_val']
        y_val = kwargs['y_val']
        name = kwargs['name']
        val_loss = self.evaluate(x_val, y_val, name)
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

    def train_now(self, x_train, y_train, x_val, y_val):
        """
        Trains the model giving useful metrics between epochs.

        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """
        print('info: training classifier...')
        import torch.optim as optim

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        dataloader = DataLoader(tuple(zip(x_train, y_train)), batch_size=self.batch_size, shuffle=False, num_workers=4)
        epoch = self.initial_epoch
        while epoch < self.epochs:  # loop over the dataset multiple times
            epoch += 1
            running_loss = 0.0
            losses = []
            for i_batch, sample_batched in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                x_i = sample_batched[0]
                y_i = sample_batched[1]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(x_i)
                loss = criterion(outputs, y_i)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                losses.append(loss)
                if i_batch % 100 == 99:  # print every 2000 mini-batches
                    print('metric: [%d, %5d] train loss: %.3f' %
                          (epoch, i_batch + 1, running_loss / 100))
                    running_loss = 0.0
            # post-epoch behaviour
            feed_dict = {'x_val': x_val,
                         'y_val': y_val,
                         'name': 'validation',
                         'losses': losses
                         }
            self.post_epoch(epoch, **feed_dict)
        print('info: finished training by batch count')

    def evaluate(self, x, y, name='test'):
        """
        Evaluate model parameters against input data.
        It logs multi-class classification performance metrics in runtime.
        :param x: Input feed tensor
        :param y: Expected abels tensor
        :param name: Evaluation title name for logs.
        :return: Total Test set Loss
        """
        print('info: evaluating classifier with {} set...'.format(name))
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(tuple(zip(x, y)), batch_size=self.batch_size, shuffle=False, num_workers=4)

        losses = []
        total = 0
        metrics = defaultdict(lambda: {'hit': 0, 'total': 0})
        for i_batch, sample_batched in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            x = sample_batched[0]
            labels = sample_batched[1]

            # forward + backward + optimize
            outputs = self(x)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            matches = (predicted == labels).squeeze()
            for idx in range(labels.size(0)):
                label = labels[idx]
                metrics[label.item()]['hit'] += matches[idx].item()
                metrics[label.item()]['total'] += 1

        evaluation_loss = reduce(lambda x, y: x + y, losses)
        print('metric: total {} loss: {}'.format(name, evaluation_loss))
        for key in metrics.keys():
            print('metric: %s accuracy of %5s : %2d %%' % (name,
                                                           key, 100 * metrics[key]['hit'] / metrics[key]['total']))
        return evaluation_loss

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
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = torch.Tensor(self.pe[:, :seq_len])
            x = x + pe
            return x


class WaveNetTransformerEncoderClassifier(TorchClassificationModel):
    model_name = 'wavenet_transformer_classif'

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

        d_model = TRANSFORMER_D_MODEL
        nhead = TRANSFORMER_N_HEAD
        num_layers = TRANSFORMER_N_LAYERS

        # reduce sample resolution from 160k to 32k
        # output_length = floor((input_length - stride)/kernel_size + 1)
        self.conv_downsampler_1 = nn.Conv1d(
            in_channels=WAVENET_END_CHANNELS,
            out_channels=128,
            kernel_size=20,
            stride=10,
            dilation=2
        )
        self.conv_downsampler_2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=20,
            stride=10,
            dilation=2
        )
        self.conv_downsampler_3 = nn.Conv1d(
            in_channels=256,
            out_channels=d_model,
            kernel_size=20,
            stride=10,
            dilation=2
        )

        self.positional_encoder = PositionalEncoder(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

        self.soft_max = nn.Softmax(dim=1)

        self.to(self.device)
        self.wavenet.to(self.device)
        self.positional_encoder.to(self.device)
        encoder_layer.to(self.device)
        self.transformer_encoder.to(self.device)

    def forward(self, x):
        print('info: feeding wavenet...')
        x = self.wavenet.forward(x)
        # reduce sequence_length / 10 three times == 16Khz to 10Hz; increase the number of channels
        x = self.conv_downsampler_1(x)
        x = self.conv_downsampler_2(x)
        x = self.conv_downsampler_3(x)
        # x.shape for convs is n_data, n_channels, n_sequence
        # transformer expected input is n_data, n_sequence, wavenet_channels
        x = x.transpose(1, 2)
        print('info: feeding transformer...')
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)  # shape  n_data, n_sequence, d_model
        x = x[:, -1, :]  # pick the last vector from the output as the sentence embedding
        # x final shape is n_data, lstm_hidden_size * 2
        print('info: feeding fully-connected...')
        # simple classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model from features from a data folder')

    parser.add_argument('--model', help='name of the model to be trained (options: ResNetV2, leglaive)',
                        default='waveNetTransformer')

    parser.add_argument('--features_path', help='Path to features folder',
                        default=FEATURES_DATA_PATH)

    parser.add_argument('--experiment', help='Name of the experiment. affects checkpoint names',
                        default='faith_tull_binary2')

    parser.add_argument('--device_name', help='Name of the device. Can be cuda:0, cuda:1, ... or cpu. '
                                              'If no device is avaiable cpu is used.',
                        default='cuda:0')

    args = parser.parse_args()
    model = args.model
    features_path = pathlib.Path(args.features_path)
    experiment_name = args.experiment
    device_name = args.device_name

    print('info: feature_path is {}'.format(features_path))
    print('info: experiment_name is {}'.format(experiment_name))
    print('info: device_name is {}'.format(device_name))

    if model == 'ResNetV2':
        train_dm, test_dm, _ = ResnetDataManager.init_n_split(
            WindowedMelSpectralCoefficientsFeatureExtractor.feature_name,
            shuffle=True,
            ratio=(0.5, 0.5, 0)
        )
        x_train = train_dm.X
        y_train = train_dm.Y
        test_dm.data_loader()
        x_test = test_dm.X
        y_test = test_dm.Y.astype(np.long)

        model = ResNetV2('faith_tull_binary_sid', num_classes=y_train.shape[1], input_shape=x_train.shape)

        model.train_now(x_train, y_train, x_test, y_test)
        model.evaluate(x_test, y_test)
    elif model == 'SimpleConvNet':
        train_dm, test_dm, val_dm = TorchVisionDataManager.init_n_split(
            WindowedMelSpectralCoefficientsFeatureExtractor.feature_name,
            shuffle=True,
            ratio=(0.5, 0.25, 0.25)
        )
        x_train = train_dm.X
        y_train = train_dm.Y
        # load validation data to ram
        val_dm.data_loader()
        x_val = val_dm.X
        y_val = val_dm.Y

        model = Simple2dConvNet('faith_tull_binary_sid', num_classes=len(set([label_id for label_id in y_train])),
                                input_shape=x_train.shape, model_type='simpleconv')

        model = model.load_checkpoint()
        # train
        model.train_now(x_train, y_train, x_val, y_val)

        # test with unseen data when finished
        # load test data to ram
        test_dm.data_loader()
        x_test = test_dm.X
        y_test = test_dm.Y
        model.evaluate(x_test, y_test)
    elif model == 'Simple1dConvNet':
        train_dataset, test_dataset, val_dataset = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (S1DCONV_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_SEQUENCE_LENGTH)
        model = Simple1dConvNet(
            'faith_tull_binary2',
            num_classes=2,
            input_shape=input_shape
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, val_dataset)
        model.evaluate(test_dataset)
    elif model == 'waveNet':
        train_dataset, test_dataset, val_dataset = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=S1DCONV_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (S1DCONV_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_SEQUENCE_LENGTH)
        model = WaveNetClassifier(
            'faith_tull_binary2',
            num_classes=2,
            input_shape=input_shape
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, val_dataset)
        model.evaluate(test_dataset)
    elif model == 'waveNetLstm':
        train_dataset, test_dataset, val_dataset, number_of_classes = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (WAVENET_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_SEQUENCE_LENGTH)
        model = WaveNetBiLSTMClassifier(
            experiment_name,
            num_classes=number_of_classes,
            input_shape=input_shape,
            device_name=device_name
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, val_dataset)
        model.evaluate(test_dataset)
    elif model == 'waveNetTransformer':
        train_dataset, test_dataset, val_dataset, number_of_classes = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # model hyper parameters should be modified in config file
        input_shape = (WAVENET_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_SEQUENCE_LENGTH)
        model = WaveNetTransformerEncoderClassifier(
            experiment_name,
            num_classes=number_of_classes,
            input_shape=input_shape,
            device_name=device_name
        )
        model = model.load_checkpoint()
        model.train_now(train_dataset, val_dataset)
        model.evaluate(test_dataset)
