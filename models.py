import argparse
import pandas as pd
import numpy as np
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

    # def train(self, audio_data, label_data, options):
    #     """
    #     Train model specified by options with given data.
    #
    #     :param audio_data: iterable reference
    #     :param label_data: iterable reference
    #     :param options: dict-like; model dependant (cnn, aidsan, etc)
    #     :return:
    #     """
    #     self.x, self.y = self.data_loader(audio_data, label_data)
    #     return

    def predict(self, audio_data, options):
        """
        Predict with given data, and options.

        :param audio_data: reference
        :param options:
        :return:
        """
        self.x = self.data_loader(audio_data)
        return


class ResNetV2(ClassificationModel):

    def __init__(self):
        """
        resnet
        """
        self.batch_size = RESNET_V2_BATCH_SIZE  # orig paper trained all networks with batch_size=128
        self.epochs = RESNET_V2_EPOCHS
        self.depth = RESNET_V2_DEPTH
        # Model name, depth and version
        self.model_type = 'ResNet%dv%d' % (self.depth, RESNET_V2_VERSION)

        # todo: de-couple data load from model vvv
        ###############################################
        self.num_classes = 2
        data_manager = ResnetDataManager(
            WindowedMelSpectralCoefficientsFeatureExtractor.feature_name,
            'manual'
        )
        data_manager.load_all(flatten=False)
        self.X, self.Y = data_manager.X, data_manager.Y
        ###############################################

        self.model, self.callbacks = self.compile_model(
            self.num_classes,
            loss='binary_crossentropy',
            model_name='bin_sid'
        )
        super().__init__(self.model_type)

    def get_shuffle_split(self, n_splits=2, test_size=0.5, train_size=0.5):
        """
        todo: Remove this
        Return a generator to get a shuffle split of the data.
        :param n_splits:
        :param test_size:
        :param train_size:
        :return: x_train, y_train, x_test, y_test
        """
        print('info: starting shuffle-split training...')
        kf = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size)
        for train_index, test_index in kf.split(self.X):
            yield self.X[train_index], self.Y[train_index], self.X[test_index], self.Y[test_index]

    def compile_model(self, num_classes=2, loss='binary_crossentropy', model_name='bin_sid', save_dir=MODELS_DATA_PATH):
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
        model_name = '%s_%s_model.{epoch:03d}.h5' % (model_name, self.model_type)
        model_checkpoint_path = str(save_dir / model_name)
        print('info: looking for checkpoints...')

        print('info: compiling model {}...'.format(model_name))
        # Input image dimensions.
        input_shape = self.X.shape[1:]

        model = resnet_v2(input_shape=input_shape, depth=self.depth, num_classes=num_classes)
        model.compile(loss=loss,
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        model.summary()
        print(self.model_type)
        # Prepare model model saving directory.
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=model_checkpoint_path,
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
                       callbacks=self.callbacks)

    def evaluate(self, x_test, y_test):
        print('info: evaluating...')
        # Score trained model.
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model from features from a data folder')

    parser.add_argument('--model', help='name of the model to be trained (options: ResNetV2, leglaive)',
                        default='ResNetV2')

    args = parser.parse_args()
    model = args.model

    if model == 'ResNetV2':
        model = ResNetV2()
        for x_train, y_train, x_test, y_test in model.get_shuffle_split():
            model.train(x_train, y_train, x_test, y_test)
            model.evaluate(x_test, y_test)
