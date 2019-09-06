from util.SST_disan.configs import cfg
from util.SST_disan.src.utils.record_log import _logger
import tensorflow as tf
from util.SST_disan.src.model.template import ModelTemplate

from util.SST_disan.src.nn_utils.integration_func import generate_embedding_mat
from util.SST_disan.src.nn_utils.nn import linear
from util.SST_disan.src.nn_utils.disan import disan


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
