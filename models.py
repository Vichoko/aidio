import argparse
import os
import shutil
import time
from math import ceil

import numpy as np
import tensorflow as tf
from keras.engine.saving import load_model

from config import MODELS_DATA_PATH, RESNET_V2_BATCH_SIZE, RESNET_V2_EPOCHS, RESNET_V2_DEPTH, \
    RESNET_V2_VERSION, ADISAN_EPOCHS, ADISAN_LOG_DIR, ADISAN_STANDBY_LOG_DIR, ADISAN_BATCH_SIZE, ADISAN_HIDDEN_UNITS, \
    ADISAN_OPTIMIZER, ADISAN_VAR_DECAY, ADISAN_WEIGHT_DECAY_FACTOR, ADISAN_DROPOUT_KEEP_PROB, ADISAN_DECAY, ADISAN_LR, \
    ADISAN_SUMMARY_PATH, ADISAN_CHECKPOINT_PATH, ADISAN_LOAD_STEP, ADISAN_LOAD_PATH, ADISAN_LOAD_MODEL, \
    ADISAN_ANSWER_DIR, ADISAN_EVAL_PERIOD, ADISAN_LOG_PERIOD, ADISAN_MAX_STEPS, ADISAN_GPU_MEM
from features import WindowedMelSpectralCoefficientsFeatureExtractor
from loaders import ResnetDataManager, DataManager


class ClassificationModel:

    def __init__(self, model_name, model_type, num_classes, input_shape, model_path, epochs,
                 batch_size,
                 **kwargs):
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


class TimeCounter:
    def __init__(self):
        self.data_round = 0
        self.epoch_time_list = []
        self.batch_time_list = []

        # run time
        self.start_time = None

    def add_start(self):
        self.start_time = time.time()

    def add_stop(self):
        assert self.start_time is not None
        self.batch_time_list.append(time.time() - self.start_time)
        self.start_time = None

    def update_data_round(self, data_round):
        if self.data_round == data_round:
            return None, None
        else:
            this_epoch_time = sum(self.batch_time_list)
            self.epoch_time_list.append(this_epoch_time)
            self.batch_time_list = []
            self.data_round = data_round
            return this_epoch_time, \
                   1.0 * sum(self.epoch_time_list) / len(self.epoch_time_list) if len(
                       self.epoch_time_list) > 0 else 0


class RecordLog:
    def __init__(self, writeToFileInterval=20, fileName='log.txt'):
        self.writeToFileInterval = writeToFileInterval
        self.waitNumToFile = self.writeToFileInterval
        buildTime = '-'.join(time.asctime(time.localtime(time.time())).strip().split(' ')[1:-1])
        buildTime = '-'.join(buildTime.split(':'))
        logFileName = buildTime  # cfg.model_name[1:] + '_' + buildTime
        self.path = os.path.join(ADISAN_LOG_DIR or ADISAN_STANDBY_LOG_DIR, logFileName + "_" + fileName)
        self.storage = []

    def add(self, content='-' * 30, ifTime=False, ifPrint=True, ifSave=True):
        # timeStr = "   ---" + str(time()) if ifTime else ''
        timeStr = "   --- " + time.asctime(time.localtime(time.time())) if ifTime else ''
        logContent = content + timeStr
        if ifPrint:
            print(logContent)
        # check save
        if ifSave:
            self.storage.append(logContent)
            self.waitNumToFile -= 1
            if self.waitNumToFile == 0:
                self.waitNumToFile = self.writeToFileInterval
                self.writeToFile()
                self.storage = []

    def writeToFile(self):
        with open(self.path, 'a', encoding='utf-8') as file:
            for ele in self.storage:
                file.write(ele + os.linesep)

    def done(self):
        self.add('Done')


_logger = RecordLog(20)


class ADiSAN(ClassificationModel):

    def __init__(self, model_name, num_classes, input_shape, model_path=MODELS_DATA_PATH,
                 epochs=ADISAN_EPOCHS, batch_size=ADISAN_BATCH_SIZE,
                 **kwargs):
        self.model_type = 'adisan'
        super().__init__(model_name, self.model_type, num_classes, input_shape, model_path, epochs, batch_size, **kwargs)
        self.time_counter = TimeCounter()

        # adisan intern params
        self.hidden_units_num = ADISAN_HIDDEN_UNITS
        self.dropout = ADISAN_DROPOUT_KEEP_PROB
        self.wd = ADISAN_WEIGHT_DECAY_FACTOR
        self.var_decay = ADISAN_VAR_DECAY
        self.mode = 'train'
        self.decay = ADISAN_DECAY
        self.optimizer = ADISAN_OPTIMIZER
        self.learning_rate = ADISAN_LR

        # initiate model
        with tf.variable_scope(self.model_type) as scope:
            self.scope = scope.name
            #        self.max_sequence_len = max_sequence_len
            self.output_class_count = self.num_classes

            self.global_step = tf.get_variable(
                'global_step',
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False
            )
            # ---- place holder -----
            self.batch_embedding_sequence = tf.placeholder(
                tf.float32,
                [None, None, None],
                name='batch_embedding_sequence'
            )  # batch_size, max_sequence_len, embedding_size

            self.batch_sequence = tf.placeholder(
                tf.float32,
                [None, None],
                name='batch_sequence'
            )  # batch_size, max_sequence_len

            self.batch_output_labels = tf.placeholder(
                tf.int32,
                [None],
                name='batch_output_labels'
            )  # integer from 0 to class_number: (batch_size)
            self.is_train = tf.placeholder(tf.bool, [], name='is_train')

            # ----------- parameters -------------
            self.hidden_units_no = self.hidden_units_num  # Hidden Units for FCNN for classification
            self.batch_size = tf.shape(self.batch_embedding_sequence)[0]
            self.max_sequence_length = tf.shape(self.batch_embedding_sequence)[1]
            self.embedding_dim = tf.shape(self.batch_embedding_sequence)[2]

            # ------------ other ---------
            # self.batch_access_mask = tf.placeholder(
            #     tf.bool,
            #     [None, None],
            #     name='batch_access_mask'
            # )  # boolean mask to ignore sequence elements, (batch_size, max_seq)
            self.batch_access_mask = tf.cast(self.batch_sequence, tf.bool)

            # self.token_mask = tf.cast(self.token_seq,
            #                           tf.bool)  # boolean mask to ignore sequence elements, (batch_size, max_seq)
            self.tensor_dict = {}  # needed for disan architecture, set 'emb' to the embedding

            # ------ start ------
            self.logits = None  # results
            self.loss = None  # loss
            self.accuracy = None
            self.var_ema = None
            self.ema = None
            self.summary = None
            self.opt = None  # optimizer (adam, adadelta, rmsprop)
            self.train_op = None  # optimizer minimize
            # fill all of these here
            self.update_tensor_add_ema_and_opt()

    def build_network(self):
        """
        Build ADiSAN + Fully-Connected NN architecture,

        :return: Reference to FCNN output.
        """
        from util.SST_disan.src.nn_utils.nn import linear
        from util.SST_disan.src.nn_utils.disan import disan
        _logger.add()
        _logger.add('building %s neural network structure...' % self.model_type)

        with tf.variable_scope('emb'):
            # get the embedding matrix
            emb = self.batch_embedding_sequence
            # here emb can me changed for whatever in theory
            self.tensor_dict['emb'] = emb

        rep = disan(
            emb,
            self.batch_access_mask,
            'DiSAN',
            self.dropout,
            self.is_train,
            self.wd, 'relu', tensor_dict=self.tensor_dict, name='')

        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(
                linear(
                    [rep],
                    self.hidden_units_no,
                    bias=True,
                    scope='pre_logits_linear',
                    wd=self.wd,
                    input_keep_prob=self.dropout,
                    is_train=self.is_train
                )
            )  # batch_size, hidden_units
            logits = linear(
                [pre_logits],
                self.output_class_count,
                bias=False,
                scope='get_output',
                wd=self.wd,
                input_keep_prob=self.dropout,
                is_train=self.is_train
            )  # batch_size, output_class_count
        _logger.done()
        return logits

    def build_loss(self):
        """
        Build Loss function
        :return: Loss
        """
        # weight_decay
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        _logger.add('regularization var num: %d' % len(reg_vars))
        _logger.add('trainable var num: %d' % len(trainable_vars))
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.batch_output_labels,
            logits=self.logits
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        """
        Calculate accurracy of given output of the network (self.logits)
        versus the expected labels (self.output_labels)
        :return: Numpy array tf.float32 (batch_size, )
        """
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.batch_output_labels
        )  # [bs]
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):
        """
        Build Network to self.logits, Loss to self.loss, Accuracy to self.accuracy.
        Set ExponentialMovingAverage in self.ema
        Set Optimizer en self.opt
        :return: None
        """
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

        # ------------ema-------------
        if True:
            self.var_ema = tf.train.ExponentialMovingAverage(self.var_decay)
            self.build_var_ema()

        if self.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(self.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if self.optimizer.lower() == 'adadelta':
            assert 0.1 < self.learning_rate < 1.
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            assert self.learning_rate < 0.1
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            assert self.learning_rate < 0.1
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % self.optimizer)

        self.train_op = self.opt.minimize(self.loss, self.global_step,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables(), )
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def step(self, sess, batch_samples, data_manager, get_summary=False):
        """
        Training step of the whole Network.

        :param sess: TF Session
        :param batch_samples: Iterator of samples with encoded data/label which should be parsed
        to TF variables in get_feed_dict
        :param get_summary: boolean flag to include summary
        :return: loss, summary and train_op session run results
        """
        assert isinstance(sess, tf.Session)
        # get embedding_sequence, output_labels and is_train flag from batch_samples
        feed_dict = data_manager.get_feed_dict(self, batch_samples, 'train')
        self.time_counter.add_start()
        if get_summary:
            loss, summary, train_op = sess.run([self.loss,
                                                self.summary,
                                                self.train_op],
                                               feed_dict=feed_dict)

        else:
            loss, train_op = sess.run([self.loss,
                                       self.train_op],
                                      feed_dict=feed_dict)
            summary = None
        self.time_counter.add_stop()
        return loss, summary, train_op

    def train(self, train_dm: DataManager, test_dm: DataManager, dev_dm: DataManager):
        """
            Legacy code to trigger the training on the given model.

        :param train_dm:
        :param test_dm:
        :param dev_dm:
        :return:
        """
        """
            :param model: ModelDiSAN instance
            :param dev_data_obj: Object with data_type attr and generate_batch_sample_iter() method
            :param test_data_obj: Object with data_type attr and generate_batch_sample_iter() method.
            :param train_data_obj: Object with sample_num attr and generate_batch_sample_iter(num_steps) method
            :return:
            """

        class GraphHandler:
            def __init__(self, model):
                """
                This class manages the model load and save the actual
                tensorflow model checkpoints and summaries
                :param model:
                """
                self.model = model
                self.saver = tf.train.Saver(max_to_keep=3)
                self.writer = None
                self.summary_path = ADISAN_SUMMARY_PATH  # str
                self.checkpoint_path = ADISAN_CHECKPOINT_PATH  # str
                self.load_step = ADISAN_LOAD_STEP  # int
                self.load_path = ADISAN_LOAD_PATH  # str specify which pre-trianed model to be load
                self.load_model = ADISAN_LOAD_MODEL
                self.mode = 'train'

            def initialize(self, sess):
                sess.run(tf.global_variables_initializer())
                if self.load_model or self.mode != 'train':
                    self.restore(sess)
                if self.mode == 'train':
                    self.writer = tf.summary.FileWriter(logdir=self.summary_path, graph=tf.get_default_graph())

            def add_summary(self, summary, global_step):
                _logger.add()
                _logger.add('saving summary...')
                self.writer.add_summary(summary, global_step)
                _logger.done()

            def add_summaries(self, summaries, global_step):
                for summary in summaries:
                    self.add_summary(summary, global_step)

            def save(self, sess, global_step=None):
                _logger.add()
                _logger.add('saving model to %s' % self.checkpoint_path)
                self.saver.save(sess, self.checkpoint_path, global_step)
                _logger.done()

            def restore(self, sess):
                _logger.add()
                # print(self.ckpt_dir)

                if self.load_step is None:
                    if self.load_path is None:
                        _logger.add('trying to restore from dir %s' % self.checkpoint_path)
                        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
                    else:
                        latest_checkpoint_path = self.load_path
                else:
                    latest_checkpoint_path = self.checkpoint_path + '-' + str(self.load_step)

                if latest_checkpoint_path is not None:
                    _logger.add('trying to restore from ckpt file %s' % latest_checkpoint_path)
                    try:
                        self.saver.restore(sess, latest_checkpoint_path)
                        _logger.add('success to restore')
                    except tf.errors.NotFoundError:
                        _logger.add('failure to restore')
                        if self.mode != 'train': raise FileNotFoundError('canot find model file')
                else:
                    _logger.add('No check point file in dir %s ' % self.checkpoint_path)
                    if self.mode != 'train': raise FileNotFoundError('canot find model file')

                _logger.done()

        class Evaluator:

            def __init__(self, model: ADiSAN):
                self.model = model
                self.global_step = self.model.global_step

                ## ---- summary----
                self.build_summary()
                self.writer = tf.summary.FileWriter(self.summary_dir)

                # config vars
                self.summary_dir = ADISAN_SUMMARY_PATH
                self.answer_dir = ADISAN_ANSWER_DIR
                self.fine_grained = NotImplemented

            def mkdir(self, *args):
                from os.path import join
                dirPath = join(*args)

                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)
                return dirPath

            # --- external use ---
            def get_evaluation(self, sess, data_manager: DataManager, global_step=None):
                """
                todo: hacer datasetobj un datamanager
                :param sess:
                :param data_manager:
                :param global_step:
                :return:
                """
                _logger.add()
                _logger.add('getting evaluation result for %s' % data_manager.data_type)

                logits_list, loss_list, accu_list = [], [], []
                is_sent_list = []
                for sample_batch, _, _, _ in data_manager.batch_iterator():
                    feed_dict = data_manager.get_feed_dict(self.model, sample_batch, 'dev')
                    logits, loss, accu = sess.run([self.model.logits,
                                                   self.model.loss,
                                                   self.model.accuracy], feed_dict)
                    logits_list.append(np.argmax(logits, -1))
                    loss_list.append(loss)
                    accu_list.append(accu)
                    is_sent_list += [sample['is_sent'] for sample in sample_batch]
                logits_array = np.concatenate(logits_list, 0)
                loss_value = np.mean(loss_list)
                accu_array = np.concatenate(accu_list, 0)
                accu_value = np.mean(accu_array)
                sent_accu_list = []
                for idx, is_sent in enumerate(is_sent_list):
                    if is_sent:
                        sent_accu_list.append(accu_array[idx])
                sent_accu_value = np.mean(sent_accu_list)

                # analysis
                # analysis_save_dir = self.mkdir(self.answer_dir,'gs_%s'%global_step or 'test')
                # OutputAnalysis.do_analysis(dataset_obj, logits_array, accu_array, analysis_save_dir,
                #                            self.fine_grained)

                # add summary
                if global_step is not None:
                    if data_manager.data_type == 'train':
                        summary_feed_dict = {
                            self.train_loss: loss_value,
                            self.train_accuracy: accu_value,
                            self.train_sent_accuracy: sent_accu_value,
                        }
                        summary = sess.run(self.train_summaries, summary_feed_dict)
                        self.writer.add_summary(summary, global_step)
                    elif data_manager.data_type == 'dev':
                        summary_feed_dict = {
                            self.dev_loss: loss_value,
                            self.dev_accuracy: accu_value,
                            self.dev_sent_accuracy: sent_accu_value,
                        }
                        summary = sess.run(self.dev_summaries, summary_feed_dict)
                        self.writer.add_summary(summary, global_step)
                    else:
                        summary_feed_dict = {
                            self.test_loss: loss_value,
                            self.test_accuracy: accu_value,
                            self.test_sent_accuracy: sent_accu_value,
                        }
                        summary = sess.run(self.test_summaries, summary_feed_dict)
                        self.writer.add_summary(summary, global_step)
                    return loss_value, accu_value, sent_accu_value

            # def get_evaluation_file_output(self, sess, data_manager, global_step, deleted_step):
            #     _logger.add()
            #     _logger.add('get evaluation file output for %s' % data_manager.data_type)
            #     # delete old file
            #     if deleted_step is not None:
            #         delete_name = 'gs_%d' % deleted_step
            #         delete_path = os.path.join(self.answer_dir, delete_name)
            #         if os.path.exists(delete_path):
            #             shutil.rmtree(delete_path)
            #         _logger.add()
            #         _logger.add('getting evaluation result for %s' % data_manager.data_type)
            #
            #     logits_list, loss_list, accu_list = [], [], []
            #     is_sent_list = []
            #     for sample_batch, _, _, _ in data_manager.generate_batch_sample_iter():
            #         feed_dict = data_manager.get_feed_dict(self.model, sample_batch, 'dev')
            #         logits, loss, accu = sess.run([self.model.logits,
            #                                        self.model.loss, self.model.accuracy], feed_dict)
            #         logits_list.append(np.argmax(logits, -1))
            #         loss_list.append(loss)
            #         accu_list.append(accu)
            #         is_sent_list += [sample['is_sent'] for sample in sample_batch]
            #     logits_array = np.concatenate(logits_list, 0)
            #     loss_value = np.mean(loss_list)
            #     accu_array = np.concatenate(accu_list, 0)
            #     accu_value = np.mean(accu_array)
            #     sent_accu_list = []
            #     for idx, is_sent in enumerate(is_sent_list):
            #         if is_sent:
            #             sent_accu_list.append(accu_array[idx])
            #     sent_accu_value = np.mean(sent_accu_list)
            #
            #     # analysis
            #     analysis_save_dir = self.mkdir(self.answer_dir, 'gs_%s' % global_step or 'test')
            #     OutputAnalysis.do_analysis(data_manager, logits_array, accu_array, analysis_save_dir,
            #                                self.fine_grained)

            # --- internal use ------
            def build_summary(self):
                with tf.name_scope('train_summaries'):
                    self.train_loss = tf.placeholder(tf.float32, [], 'train_loss')
                    self.train_accuracy = tf.placeholder(tf.float32, [], 'train_accuracy')
                    self.train_sent_accuracy = tf.placeholder(tf.float32, [], 'train_sent_accuracy')
                    tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss', self.train_loss))
                    tf.add_to_collection('train_summaries_collection',
                                         tf.summary.scalar('train_accuracy', self.train_accuracy))
                    tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_sent_accuracy',
                                                                                         self.train_sent_accuracy))
                    self.train_summaries = tf.summary.merge_all('train_summaries_collection')

                with tf.name_scope('dev_summaries'):
                    self.dev_loss = tf.placeholder(tf.float32, [], 'dev_loss')
                    self.dev_accuracy = tf.placeholder(tf.float32, [], 'dev_accuracy')
                    self.dev_sent_accuracy = tf.placeholder(tf.float32, [], 'dev_sent_accuracy')
                    tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss', self.dev_loss))
                    tf.add_to_collection('dev_summaries_collection',
                                         tf.summary.scalar('dev_accuracy', self.dev_accuracy))
                    tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_sent_accuracy',
                                                                                       self.dev_sent_accuracy))
                    self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')

                with tf.name_scope('test_summaries'):
                    self.test_loss = tf.placeholder(tf.float32, [], 'test_loss')
                    self.test_accuracy = tf.placeholder(tf.float32, [], 'test_accuracy')
                    self.test_sent_accuracy = tf.placeholder(tf.float32, [], 'test_sent_accuracy')
                    tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss', self.test_loss))
                    tf.add_to_collection('test_summaries_collection',
                                         tf.summary.scalar('test_accuracy', self.test_accuracy))
                    tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_sent_accuracy',
                                                                                        self.test_sent_accuracy))
                    self.test_summaries = tf.summary.merge_all('test_summaries_collection')

        # from util.SST_disan.src.perform_recorder import PerformRecoder
        graphHandler = GraphHandler(self)
        evaluator = Evaluator(self)
        # performRecoder = PerformRecoder(3)
        if ADISAN_GPU_MEM < 1.:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=ADISAN_GPU_MEM,
                                        allow_growth=True)
        else:
            gpu_options = tf.GPUOptions()
        graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.Session(config=graph_config)
        graphHandler.initialize(sess)
        # begin training
        steps_per_epoch = ceil(1.0 * train_dm.sample_num / train_dm.batch_size)
        num_steps = ADISAN_MAX_STEPS or steps_per_epoch * ADISAN_EPOCHS

        for sample_batch, batch_num, data_round, idx_b in train_dm.batch_iterator(num_steps):
            global_step = sess.run(model.global_step) + 1
            if_get_summary = global_step % (ADISAN_LOG_PERIOD or steps_per_epoch) == 0
            loss, summary, train_op = model.step(
                sess,
                sample_batch,
                get_summary=if_get_summary
            )

            if global_step % 100 == 0:
                _logger.add('data round: %d: %d/%d, global step:%d -- loss: %.4f' %
                            (data_round, idx_b, batch_num, global_step, loss))

            if if_get_summary:
                graphHandler.add_summary(summary, global_step)

            # Occasional evaluation
            if global_step % (ADISAN_EVAL_PERIOD or steps_per_epoch) == 0:
                # ---- dev ----
                dev_loss, dev_accu, dev_sent_accu = evaluator.get_evaluation(
                    sess, dev_dm, global_step
                )
                _logger.add('==> for dev, loss: %.4f, accuracy: %.4f, sentence accuracy: %.4f' %
                            (dev_loss, dev_accu, dev_sent_accu))
                # ---- test ----
                test_loss, test_accu, test_sent_accu = evaluator.get_evaluation(
                    sess, test_dm, global_step
                )
                _logger.add('~~> for test, loss: %.4f, accuracy: %.4f, sentence accuracy: %.4f' %
                            (test_loss, test_accu, test_sent_accu))

                # is_in_top, deleted_step = performRecoder.update_top_list(global_step, dev_accu, sess)
                # if is_in_top and global_step > 30000:  # todo-ed: delete me to run normally
                #     # evaluator.get_evaluation_file_output(sess, dev_data_obj, global_step, deleted_step)
                #     evaluator.get_evaluation_file_output(sess, test_dm, global_step, deleted_step)
            this_epoch_time, mean_epoch_time = self.time_counter.update_data_round(data_round)
            if this_epoch_time is not None and mean_epoch_time is not None:
                _logger.add('##> this epoch time: %f, mean epoch time: %f' % (this_epoch_time, mean_epoch_time))


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
