# import keras
# from keras.layers import Dense, Conv2D, BatchNormalization, Activation
# from keras.layers import AveragePooling2D, Input, Flatten
# from keras.regularizers import l2
# from keras.models import Model
print('info: importing libs...')
import pandas as pd
import numpy as np

from config import FEATURES_DATA_PATH
from features import WindowedMelSpectralCoefficientsFeatureExtractor

from sklearn import svm, metrics
from sklearn.model_selection import ShuffleSplit

# def resnet_layer(inputs,
#                  num_filters=16,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True):
#     """2D Convolution-Batch Normalization-Activation stack builder
#
#     # Arguments
#         inputs (tensor): input tensor from input image or previous layer
#         num_filters (int): Conv2D number of filters
#         kernel_size (int): Conv2D square kernel dimensions
#         strides (int): Conv2D square stride dimensions
#         activation (string): activation name
#         batch_normalization (bool): whether to include batch normalization
#         conv_first (bool): conv-bn-activation (True) or
#             bn-activation-conv (False)
#
#     # Returns
#         x (tensor): tensor as input to the next layer
#     """
#     conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-4))
#
#     x = inputs
#     if conv_first:
#         x = conv(x)
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#     else:
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#         x = conv(x)
#     return x
#
#
# def resnet_v2(input_shape, depth, num_classes=10):
#     """ResNet Version 2 Model builder [b]
#
#     Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
#     bottleneck layer
#     First shortcut connection per layer is 1 x 1 Conv2D.
#     Second and onwards shortcut connection is identity.
#     At the beginning of each stage, the feature map size is halved (downsampled)
#     by a convolutional layer with strides=2, while the number of filter maps is
#     doubled. Within each stage, the layers have the same number filters and the
#     same filter map sizes.
#     Features maps sizes:
#     conv1  : 32x32,  16
#     stage 0: 32x32,  64
#     stage 1: 16x16, 128
#     stage 2:  8x8,  256
#
#     # Arguments
#         input_shape (tensor): shape of input image tensor
#         depth (int): number of core convolutional layers
#         num_classes (int): number of classes (CIFAR10 has 10)
#
#     # Returns
#         model (Model): Keras model instance
#     """
#     if (depth - 2) % 9 != 0:
#         raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
#     # Start model definition.
#     num_filters_in = 16
#     num_res_blocks = int((depth - 2) / 9)
#
#     inputs = Input(shape=input_shape)
#     # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
#     x = resnet_layer(inputs=inputs,
#                      num_filters=num_filters_in,
#                      conv_first=True)
#
#     # Instantiate the stack of residual units
#     for stage in range(3):
#         for res_block in range(num_res_blocks):
#             activation = 'relu'
#             batch_normalization = True
#             strides = 1
#             if stage == 0:
#                 num_filters_out = num_filters_in * 4
#                 if res_block == 0:  # first layer and first stage
#                     activation = None
#                     batch_normalization = False
#             else:
#                 num_filters_out = num_filters_in * 2
#                 if res_block == 0:  # first layer but not first stage
#                     strides = 2  # downsample
#
#             # bottleneck residual unit
#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters_in,
#                              kernel_size=1,
#                              strides=strides,
#                              activation=activation,
#                              batch_normalization=batch_normalization,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_in,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_out,
#                              kernel_size=1,
#                              conv_first=False)
#             if res_block == 0:
#                 # linear projection residual shortcut connection to match
#                 # changed dims
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters_out,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = keras.layers.add([x, y])
#
#         num_filters_in = num_filters_out
#
#     # Add classifier on top.
#     # v2 has BN-ReLU before Pooling
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(num_classes,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)
#
#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

print('info: setting params...')
# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = False
num_classes = 2
depth = 3 * 9 + 2
flatten = True
# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, 2)

feature_name = WindowedMelSpectralCoefficientsFeatureExtractor.feature_name
feature_data_path = FEATURES_DATA_PATH / feature_name
print('info: loading data from disk...')
labels_df = pd.read_csv(
     feature_data_path /
    'labels.{}.csv'.format(feature_name)
)
filenames = labels_df['filename']
labels = labels_df['label']
X = np.asarray([np.load(feature_data_path / filename) for filename in filenames])
Y = np.asarray(labels)
print('info: formatting data...')
X = X.reshape((X.shape[0], -1)) if flatten else X
print('info: starting shuffle-split training...')
kf = ShuffleSplit(n_splits=2, test_size=0.5, train_size=0.5)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]
    print('info: initiating classifier...')
    classifier = svm.SVC(gamma=0.001)
    print('info: training classifier...')
    classifier.fit(x_train, y_train)
    print('info: predicting...')
    y_pred = classifier.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))

print('info: done')
