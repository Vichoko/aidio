import argparse
import os
from collections import defaultdict
from functools import reduce
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.engine.saving import load_model
from torch.utils.data.dataloader import DataLoader

from config import MODELS_DATA_PATH, RESNET_V2_BATCH_SIZE, RESNET_V2_EPOCHS, RESNET_V2_DEPTH, \
    RESNET_V2_VERSION, SIMPLECONV_BATCH_SIZE, SIMPLECONV_EPOCHS
from features import WindowedMelSpectralCoefficientsFeatureExtractor
from loaders import ResnetDataManager, TorchVisionDataManager


class ClassificationModel:

    def __init__(self, model_name, model_type, num_classes, input_shape, model_path, epochs,
                 batch_size,
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
        model_filename = '%s_%s_model.{epoch:03d}.h5' % (self.model_name, self.model_type)
        self.model_checkpoint_path = model_path / model_filename
        self.batch_size = batch_size
        self.initial_epoch = 0

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


class TorchClassificationModel(nn.Module):

    def __init__(self, model_name, model_type, num_classes, input_shape, model_path, epochs,
                 batch_size,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.epochs = epochs
        model_filename = '%s_%s_model.{epoch:03d}.h5' % (self.model_name, self.model_type)
        self.model_checkpoint_path = model_path / model_filename
        self.batch_size = batch_size
        self.initial_epoch = 0


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

    def train_now(self, x_train, y_train, x_val, y_val):
        x_train = x_train.numpy() if torch.is_tensor(x_train) else x_train
        y_train = y_train.numpy() if torch.is_tensor(y_train) else y_train
        x_val = x_val.numpy() if torch.is_tensor(x_val) else x_val
        y_val = y_val.numpy() if torch.is_tensor(y_val) else y_val

        print('info: training classifier...')
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(x_val, y_val),
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


class SimpleConvNet(ClassificationModel, nn.Module):
    model_name = 'simpleConvNet'

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model from features from a data folder')

    parser.add_argument('--model', help='name of the model to be trained (options: ResNetV2, leglaive)',
                        default='SimpleConvNet')

    args = parser.parse_args()
    model = args.model

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

        model = SimpleConvNet('faith_tull_binary_sid', num_classes=len(set([label_id for label_id in y_train])),
                              input_shape=x_train.shape, model_type='simpleconv')

        # check for existing checkpoints
        chkp_files, chkp_epoch = model.checkpoint_files
        assert len(chkp_files) == len(chkp_epoch)
        if len(chkp_files) != 0:
            # load from checkpoint
            max_idx = int(np.argmax(chkp_epoch))
            model = torch.load(str(chkp_files[max_idx]))
            model.initial_epoch = int(chkp_epoch[max_idx])
            print('info: loading model from checkpoint {}'.format(chkp_files[max_idx]))
        # train
        model.train_now(x_train, y_train, x_val, y_val)

        # test with unseen data when finished
        # load test data to ram
        test_dm.data_loader()
        x_test = test_dm.X
        y_test = test_dm.Y
        model.evaluate(x_test, y_test)
