from util.SST_disan.configs import cfg
from util.SST_disan.src.utils.record_log import _logger
import tensorflow as tf


class GraphHandler(object):
    def __init__(self, model):
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=3)
        self.writer = None
        self.load_model = cfg.load_model
        self.mode = cfg.mode
        self.summary_dir = cfg.summary_dir
        self.ckpt_path = cfg.ckpt_path
        self.load_step = cfg.load_step
        self.ckpt_dir = cfg.load_path
        self.load_path = cfg.load_path

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        if self.load_model or self.mode != 'train':
            self.restore(sess)
        if self.mode == 'train':
            self.writer = tf.summary.FileWriter(logdir=self.summary_dir, graph=tf.get_default_graph())

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
        _logger.add('saving model to %s' % self.ckpt_path)
        self.saver.save(sess, self.ckpt_path, global_step)
        _logger.done()

    def restore(self, sess):
        _logger.add()
        # print(self.ckpt_dir)

        if self.load_step is None:
            if self.load_path is None:
                _logger.add('trying to restore from dir %s' % self.ckpt_dir)
                latest_checkpoint_path = tf.train.latest_checkpoint(self.ckpt_dir)
            else:
                latest_checkpoint_path = self.load_path
        else:
            latest_checkpoint_path = self.ckpt_path + '-' + str(self.load_step)

        if latest_checkpoint_path is not None:
            _logger.add('trying to restore from ckpt file %s' % latest_checkpoint_path)
            try:
                self.saver.restore(sess, latest_checkpoint_path)
                _logger.add('success to restore')
            except tf.errors.NotFoundError:
                _logger.add('failure to restore')
                if self.mode != 'train': raise FileNotFoundError('canot find model file')
        else:
            _logger.add('No check point file in dir %s ' % self.ckpt_dir)
            if self.mode != 'train': raise FileNotFoundError('canot find model file')

        _logger.done()
