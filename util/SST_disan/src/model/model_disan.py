from util.SST_disan.configs import cfg
from util.SST_disan.src.utils.record_log import _logger
import tensorflow as tf
from util.SST_disan.src.model.template import ModelTemplate

from util.SST_disan.src.nn_utils.integration_func import generate_embedding_mat
from util.SST_disan.src.nn_utils.nn import linear
from util.SST_disan.src.nn_utils.disan import disan


class ModelADiSAN:
    def __init__(self, output_class_count, scope):
        self.scope = scope
        #        self.max_sequence_len = max_sequence_len
        self.output_class_count = output_class_count

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

        self.batch_output_labels = tf.placeholder(
            tf.int32,
            [None],
            name='batch_output_labels'
        )  # integer from 0 to class_number: (batch_size)
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # ----------- parameters -------------
        self.hidden_units_no = cfg.hidden_units_num  # Hidden Units for FCNN for classification
        self.finetune_emb = cfg.fine_tune  # ???
        self.batch_size = tf.shape(self.batch_embedding_sequence)[0]
        self.max_sequence_length = tf.shape(self.batch_embedding_sequence)[1]
        self.embedding_dim = tf.shape(self.batch_embedding_sequence)[2]

        # ------------ other ---------
        self.batch_access_mask = tf.placeholder(
            tf.bool,
            [self.batch_size, self.max_sequence_length],
            name='batch_access_mask'
        )  # boolean mask to ignore sequence elements, (batch_size, max_seq)

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
        _logger.add()
        _logger.add('building %s neural network structure...' % cfg.network_type)

        with tf.variable_scope('emb'):
            # get the embedding matrix
            emb = self.batch_embedding_sequence
            # here emb can me changed for whatever in theory
            self.tensor_dict['emb'] = emb

        rep = disan(
            emb,
            self.batch_access_mask,
            'DiSAN',
            cfg.dropout,
            self.is_train,
            cfg.wd, 'relu', tensor_dict=self.tensor_dict, name='')

        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(
                linear(
                    [rep],
                    self.hidden_units_no,
                    bias=True,
                    scope='pre_logits_linear',
                    wd=cfg.wd,
                    input_keep_prob=cfg.dropout,
                    is_train=self.is_train
                )
            )  # batch_size, hidden_units
            logits = linear(
                [pre_logits],
                self.output_class_count,
                bias=False,
                scope='get_output',
                wd=cfg.wd,
                input_keep_prob=cfg.dropout,
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
                weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.wd,
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
            self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
            self.build_var_ema()

        if cfg.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if cfg.optimizer.lower() == 'adadelta':
            assert 0.1 < cfg.learning_rate < 1.
            self.opt = tf.train.AdadeltaOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            assert cfg.learning_rate < 0.1
            self.opt = tf.train.AdamOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            assert cfg.learning_rate < 0.1
            self.opt = tf.train.RMSPropOptimizer(cfg.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)

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

    def get_feed_dict(self, sample_batch, data_type='train'):
        """
        Instance tf.variable values from sample_batch.

        This method unify the parsing of the custom data to a standarized input for the NN.

        The returned feed_dict should include:
            @deprecated: self.token_seq: index of embedding: batch_size, max_length
            self.embedding_seq: sequence embeddings # batch_size, max_sequence_len, embedding_size
            self.output_labels integer from 0 to class_number: (batch_size)
            self.is_train True or False depending if it's training


        :param sample_batch: Iterator of training examples with their labels.
        :param data_type: String flag to tell if training or not
        :return: feed_dict with gathered values
        """

        feed_dict = {self.batch_embedding_sequence: None,
                     self.batch_output_labels: None,
                     self.batch_access_mask: None,
                     self.is_train: True if data_type == 'train' else False}
        return feed_dict

    def step(self, sess, batch_samples, get_summary=False):
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
        feed_dict = self.get_feed_dict(batch_samples, 'train')
        cfg.time_counter.add_start()
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
        cfg.time_counter.add_stop()
        return loss, summary, train_op


class ModelDiSAN(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        super(ModelDiSAN, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        _logger.add()
        _logger.add('building %s neural network structure...' % cfg.network_type)

        with tf.variable_scope('emb'):
            # get the embedding matrix for query by tf.nn.embedding_lookup
            token_emb_mat = generate_embedding_mat(self.token_set_len, self.word_embedding_len,
                                                   init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat,
                                                   extra_trainable=self.finetune_emb,  # is false
                                                   scope='gene_token_emb_mat')
            # lookup from token_ids to embeddings
            emb = tf.nn.embedding_lookup(token_emb_mat,
                                         self.token_seq
                                         )  # batch_size,max_sequence_length,word_embedding_len
            # here emb can me changed for whatever in theory
            self.tensor_dict['emb'] = emb

        rep = disan(
            emb,
            self.token_mask,
            'DiSAN',
            cfg.dropout,
            self.is_train,
            cfg.wd, 'relu', tensor_dict=self.tensor_dict, name='')

        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(
                linear(
                    [rep],
                    self.hidden_units_no,
                    bias=True,
                    scope='pre_logits_linear',
                    wd=cfg.wd,
                    input_keep_prob=cfg.dropout,
                    is_train=self.is_train
                )
            )  # batch_size, hidden_units
            logits = linear(
                [pre_logits],
                self.output_class_count,
                bias=False,
                scope='get_output',
                wd=cfg.wd,
                input_keep_prob=cfg.dropout,
                is_train=self.is_train
            )  # batch_size, 5 (output_classes)
        _logger.done()
        return logits
