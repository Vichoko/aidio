from util.SST_disan.configs import cfg
from util.SST_disan.src.utils.record_log import _logger
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        """

        :param token_emb_mat: (#_tokens, 300) matrix with embeddings indexed by token set order
        :param glove_emb_mat: (#_glove_vectors, 300) matrix with embeddings indexed by glove token set order
        :param tds: token set length
        :param cds: char set length
        :param tl: max token length
        :param scope: scope name
        """
        self.scope = scope
        self.global_step = tf.get_variable(
            'global_step',
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        # custom case variables
        self.token_emb_mat, self.glove_emb_mat = token_emb_mat, glove_emb_mat

        # ---- place holder -----
        # todo: remove token_seq as audio_token is poorly defined
        self.token_seq = tf.placeholder(
            tf.int32,
            [None, None],
            name='token_seq'
        )  # index of emb: batch_size, max_length
        self.embedding_seq = tf.placeholder(
            tf.float32,
            [None, None, None],
            name='embedding_seq'
        )  # batch_size, max_sequence_len, embedding_size

        self.output_labels = tf.placeholder(
            tf.int32,
            [None],
            name='output_labels'
        )  # integer from 0 to class_number: (batch_size)
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # ----------- parameters -------------
        self.token_set_len = tds
        self.max_token_len = tl
        self.word_embedding_len = cfg.word_embedding_length
        # self.char_embedding_len = cfg.char_embedding_length
        # self.char_out_size = cfg.char_out_size
        # self.out_channel_dims = list(map(int, cfg.out_channel_dims.split(',')))
        # self.filter_height = list(map(int, cfg.filter_heights.split(',')))
        self.hidden_units_no = cfg.hidden_units_num
        self.finetune_emb = cfg.fine_tune

        self.output_class_count = 5 if cfg.fine_grained else 2

        self.batch_size = tf.shape(self.token_seq)[0]
        # self.max_sequence_len = tf.shape(self.token_seq)[1]

        # ------------ other ---------
        self.token_mask = tf.cast(self.token_seq, tf.bool)
        # self.token_len = tf.reduce_sum(tf.cast(self.token_mask, tf.int32), -1)
        self.tensor_dict = {}

        # ------ start ------
        self.logits = None  # results
        self.loss = None  # loss
        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.summary = None
        self.opt = None  # optimizer (adam, adadelta, rmsprop)
        self.train_op = None  # optimizer minimize

    @abstractmethod
    def build_network(self):
        pass

    def build_loss(self):
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
            labels=self.output_labels,
            logits=self.logits
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.output_labels
        )  # [bs]
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):
        """
        Build Network in self.logits, Loss in self.loss, Accuracy in self.accuracy.
        Set ExponentialMovingAverage in self.ema
        Set Optimizer en self.opt
        :return:
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
            assert cfg.learning_rate > 0.1 and cfg.learning_rate < 1.
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
        # max lens
        sl, ol, mc = 0, 0, 0
        for sample in sample_batch:
            sl = max(sl, len(sample['root_node']['token_seq']))
            ol = max(ol, len(sample['shift_reduce_info']['op_list']))
            for reduce_list in sample['shift_reduce_info']['reduce_mat']:
                mc = max(mc, len(reduce_list))

        assert mc == 0 or mc == 2, mc

        # token and char
        token_seq_b = []
        char_seq_b = []
        for sample in sample_batch:
            token_seq = np.zeros([sl], cfg.intX)
            char_seq = np.zeros([sl, self.max_token_len], cfg.intX)

            for idx_t, (token, char_seq_v) in enumerate(zip(sample['root_node']['token_seq_digital'],
                                                            sample['root_node']['char_seq_digital'])):
                token_seq[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c >= self.max_token_len: break
                    char_seq[idx_t, idx_c] = char
            token_seq_b.append(token_seq)
            char_seq_b.append(char_seq)
        token_seq_b = np.stack(token_seq_b)
        char_seq_b = np.stack(char_seq_b)

        # label
        sentiment_label_b = []
        for sample in sample_batch:
            sentiment_float = sample['root_node']['sentiment_label']
            sentiment_int = cfg.sentiment_float_to_int(sentiment_float)
            sentiment_label_b.append(sentiment_int)
        sentiment_label_b = np.stack(sentiment_label_b).astype(cfg.intX)

        feed_dict = {self.token_seq: token_seq_b,
                     self.output_labels: sentiment_label_b,
                     self.is_train: True if data_type == 'train' else False}
        return feed_dict

    def step(self, sess, batch_samples, get_summary=False):
        """
        Training step
        :param sess: TF Session
        :param batch_samples: Iterator of samples with encoded data/label
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
