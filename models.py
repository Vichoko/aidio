import argparse
import os
import pandas as pd
import numpy as np
from keras.engine.saving import load_model

from sklearn.model_selection import ShuffleSplit

from config import FEATURES_DATA_PATH, MODELS_DATA_PATH, RESNET_V2_BATCH_SIZE, RESNET_V2_EPOCHS, RESNET_V2_DEPTH, \
    RESNET_V2_VERSION
from features import WindowedMelSpectralCoefficientsFeatureExtractor
from loaders import ResnetDataManager, DataManager


class ClassificationModel:
    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        return

    def data_loader(self, audio_data, label_data=None):
        raise NotImplemented


class ResNetV2(ClassificationModel):

    def __init__(self, model_name, num_classes, input_shape, model_path=MODELS_DATA_PATH, epochs=RESNET_V2_EPOCHS,
                 **kwargs):
        """
        resnet
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.epochs = epochs
        self.initial_epoch = 0
        self.batch_size = RESNET_V2_BATCH_SIZE  # orig paper trained all networks with batch_size=128
        self.depth = RESNET_V2_DEPTH
        # Model name, depth and version
        self.model_name = model_name
        self.model_type = 'ResNet%dv%d' % (self.depth, RESNET_V2_VERSION)

        if self.num_classes > 2:
            self.loss = 'categorical_crossentropy'
        elif self.num_classes == 2:
            self.loss = 'binary_crossentropy'
        else:
            raise Exception('num_classes cant be lesser than 2...')

        # checkpoint save system
        model_name = '%s_%s_model.{epoch:03d}.h5' % (self.model_name, self.model_type)
        self.model_checkpoint_path = model_path / model_name

        self.model, self.callbacks = self.compile_model(
            **kwargs
        )
        super().__init__(self.model_type)

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

    def compile_model(self, **kwargs):
        """
        Compile the model to be ready to be trained.
        :return:
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

    def train(self, x_train, y_train, x_test, y_test):
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

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model from features from a data folder')

    parser.add_argument('--model', help='name of the model to be trained (options: ResNetV2, leglaive)',
                        default='ResNetV2')

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
        y_test = test_dm.Y

        model = ResNetV2('faith_tull_binary_sid', num_classes=y_train.shape[1], input_shape=x_train.shape)

        model.train(x_train, y_train, x_test, y_test)
        model.evaluate(x_test, y_test)
