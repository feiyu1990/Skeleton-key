from __future__ import division

import tensorflow as tf
import h5py
import numpy as np
from resnet import *


class Level1Model(object):
    def __init__(self, word_to_idx, dim_feature=[49, 512], dim_embed=512, dim_hidden=1800,
                 n_time_step=16, alpha_c=0.0, dropout=True, h5_name='./model/best_2level_lm_param.h5', train=True):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.alpha_c = alpha_c
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['START']
        self._null = word_to_idx['NULL']
        self._eos = word_to_idx['EOS']
        self.loss = 0.0
        self.train = train

        self.resnet = ResNet()

        # Place holder for features and captions
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.c_feed = tf.placeholder(tf.float32, [None, self.H])
        self.h_feed = tf.placeholder(tf.float32, [None, self.H])
        self.in_word = tf.placeholder(tf.int32, [None])
        self.word_feed = tf.placeholder(tf.int32, [None])

        self.init_c = None
        self.init_h = None
        self.features_proj = None
        self.features_encode = None
        self.log_softmax = None
        self.c = None
        self.h = None
        self.log_softmax0 = None
        self.c0 = None
        self.h0 = None

        self.alpha0 = None
        self.context4next0 = None
        self.context4next = None
        self.embed4next = None

        self.model_load = h5py.File(h5_name)

        w_numpy_pre = self.model_load['/lookup_table/weight'][:]
        w_numpy = np.zeros((w_numpy_pre.shape[0] + 2, w_numpy_pre.shape[1]))
        w_numpy[1:-1, :] = w_numpy_pre
        self.w_numpy = w_numpy.astype(np.float32)

    def _get_initial_lstm(self, features):
        with tf.variable_scope('level1/initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w2_init = np.transpose(self.model_load['/init_network/weight2'][:], (1, 0))
            b2_init = self.model_load['/init_network/bias2'][:]

            w_1_ = np.transpose(self.model_load['/init_network/weight1'][:], (1, 0))
            w_1 = tf.get_variable('w_w1', initializer=w_1_)
            b_1 = tf.get_variable('w_b1', initializer=self.model_load['/init_network/bias1'][:])
            h1 = tf.nn.relu(tf.matmul(features_mean, w_1) + b_1)
            # todo: this dropout can be added later
            # if self.dropout:
            # h1 = tf.nn.dropout(h1, 0.5)

            w_h = tf.get_variable('w_h', initializer=w2_init[:, self.H:])
            b_h = tf.get_variable('b_h', initializer=b2_init[self.H:])
            h = tf.nn.tanh(tf.matmul(h1, w_h) + b_h)

            w_c = tf.get_variable('w_c', initializer=w2_init[:, :self.H])
            b_c = tf.get_variable('b_c', initializer=b2_init[:self.H])
            c = tf.nn.tanh(tf.matmul(h1, w_c) + b_c)

            return c, h

    def _cnn_encoding(self, features):
        with tf.variable_scope('level1/cnn_encoding'):
            weights = np.transpose(self.model_load['/cnn_encoding/weight'][:], (2, 3, 1, 0))
            conv = tf.nn.conv2d(features, weights, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, self.model_load['/cnn_encoding/bias'][:])
            conv_reshape = tf.reshape(conv, [-1, self.L, self.D])  # (-1, 49, 512)
            return conv_reshape

    def _word_embedding(self, inputs, reuse=False):
        # todo: word_embedding for <START> token is all-zero
        with tf.variable_scope('level1/word_embedding', reuse=reuse):
            w = tf.get_variable('w', initializer=self.w_numpy)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('level1/project_features'):
            # features_proj --> proj_ctx
            # todo: features_proj = tf.matmul(features_flat, w) + b

            w1_ = np.transpose(self.model_load['/core/context_proj1/weight'][:], (1, 0))
            b1_ = self.model_load['/core/context_proj1/bias'][:]
            w2_ = np.transpose(self.model_load['/core/context_proj2/weight'][:], (1, 0))
            b2_ = self.model_load['/core/context_proj2/bias'][:]

            w1 = tf.get_variable('w1', initializer=w1_)
            b1 = tf.get_variable('b1', initializer=b1_)
            w2 = tf.get_variable('w2', initializer=w2_)
            b2 = tf.get_variable('b2', initializer=b2_)

            features_flat = tf.reshape(features, [-1, self.D])
            features_proj1 = tf.nn.tanh(tf.matmul(features_flat, w1) + b1)
            features_proj = tf.matmul(features_proj1, w2) + b2
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('level1/attention_layer', reuse=reuse):
            w_ = np.transpose(self.model_load['/core/hidden2context/weight'][:], (1, 0))
            b_ = self.model_load['/core/hidden2context/bias'][:]
            w_att_ = np.transpose(self.model_load['/core/pctx2alpha/weight'][:], (1, 0))
            b_att_ = self.model_load['/core/pctx2alpha/bias'][:]

            w = tf.get_variable('w', initializer=w_)
            b = tf.get_variable('b', initializer=b_)
            w_att = tf.get_variable('w_att', initializer=w_att_)
            b_att = tf.get_variable('b_att', initializer=b_att_)

            # tf.matmul(h, w) --> p_hidden
            # tf.expand_dims(tf.matmul(h, w), 1)) --> p_hidden_rep
            # (features_proj + tf.expand_dims(tf.matmul(h, w), 1))  --> proj_ctx_cond
            # todo: should use tanh not relu if want to match original model
            # todo: tf.expand_dims(tf.matmul(h, w) + b
            # h_att --> proj_ctx_cond_acti
            h_att = tf.nn.tanh(features_proj + tf.expand_dims(tf.matmul(h, w) + b, 1))

            # tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att) --> alpha_pre
            # out_att --> alpha_squeeze
            # alpha --> alpha
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att) + b_att, [-1, self.L])
            alpha = tf.nn.softmax(out_att)

            # features * tf.expand_dims(alpha, 2) --> z_pre
            # context --> z
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
            return context, alpha

    def _lstm(self, input_h, input_c, input_x, context, reuse=False):
        with tf.variable_scope('level1/lstm', reuse=reuse):
            w_i2h_ = np.transpose(self.model_load['/core/i2h/weight'][:], (1, 0))
            b_i2h_ = self.model_load['/core/i2h/bias'][:]
            w_h2h_ = np.transpose(self.model_load['/core/h2h/weight'][:], (1, 0))
            b_h2h_ = self.model_load['/core/h2h/bias'][:]
            w_z2h_ = np.transpose(self.model_load['/core/z2h/weight'][:], (1, 0))
            b_z2h_ = self.model_load['/core/z2h/bias'][:]

            w_i2h = tf.get_variable('w_i2h', initializer=w_i2h_)
            b_i2h = tf.get_variable('b_i2h', initializer=b_i2h_)
            w_h2h = tf.get_variable('w_h2h', initializer=w_h2h_)
            b_h2h = tf.get_variable('b_h2h', initializer=b_h2h_)
            w_z2h = tf.get_variable('w_z2h', initializer=w_z2h_)
            b_z2h = tf.get_variable('b_z2h', initializer=b_z2h_)

            input_x = tf.cast(input_x, tf.float32)
            i2h = tf.matmul(input_x, w_i2h) + b_i2h
            h2h = tf.matmul(input_h, w_h2h) + b_h2h
            z2h = tf.matmul(context, w_z2h) + b_z2h
            all_input_sums = i2h + h2h + z2h
            reshaped = tf.reshape(all_input_sums, [-1, 4, self.H])
            n1, n2, n3, n4 = tf.unstack(reshaped, axis=1)
            in_gate = tf.sigmoid(n1)
            forget_gate = tf.sigmoid(n2)
            out_gate = tf.sigmoid(n3)
            in_transform = tf.tanh(n4)
            c = tf.multiply(forget_gate, input_c) + tf.multiply(in_gate, in_transform)
            h = tf.multiply(out_gate, tf.tanh(c))
            return c, h

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('level1/selector', reuse=reuse):
            w_ = np.transpose(self.model_load['/core/h2sel/weight'][:], (1, 0))
            b_ = self.model_load['/core/h2sel/bias'][:]

            w = tf.get_variable('w', initializer=w_)
            b = tf.get_variable('b', initializer=b_)

            # tf.matmul(h, w) + b --> beta_pre
            # beta --> beta
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')

            # context --> z_select
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('level1/logits', reuse=reuse):
            x = tf.cast(x, tf.float32)
            w_h_ = np.transpose(self.model_load['/core/decoder_h/weight'][:], (1, 0))
            b_h_ = self.model_load['/core/decoder_h/bias'][:]

            b_h = tf.get_variable('b_h', initializer=b_h_)
            w_h = tf.get_variable('w_h', initializer=w_h_)

            w_out_ = np.transpose(self.model_load['/core/decoder/weight'][:], (1, 0))
            b_out_ = self.model_load['/core/decoder/bias'][:]
            w_out = tf.get_variable('w_out', initializer=w_out_)
            b_out = tf.get_variable('b_out', initializer=b_out_)

            w_ctx_ = np.transpose(self.model_load['/core/decoder_ctx/weight'][:], (1, 0))
            b_ctx_ = self.model_load['/core/decoder_ctx/bias'][:]
            w_ctx = tf.get_variable('w_ctx', initializer=w_ctx_)
            b_ctx = tf.get_variable('b_ctx', initializer=b_ctx_)

            # if dropout:
                # h --> top_h
                # h = tf.nn.dropout(h, 0.5)

            # h_logits --> h_out
            h_logits = tf.matmul(h, w_h) + b_h

            # todo: b_ctx2out
            # w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
            h_logits += tf.matmul(context, w_ctx) + b_ctx

            # out_sum --> h_logits
            h_logits += x

            # h_logits --> out_sum_acti
            h_logits = tf.nn.tanh(h_logits)

            # if dropout:
                # h_logits --> out_sum_dropout
                # h_logits = tf.nn.dropout(h_logits, 0.5)

            # out_logits --> proj
            out_logits = tf.matmul(h_logits, w_out) + b_out

            return out_logits

    def init_inference(self):
        self.resnet.build_model(is_training=self.train)
        self.features_encode = self._cnn_encoding(features=self.resnet.features)
        self.init_c, self.init_h = self._get_initial_lstm(features=self.features_encode)
        self.features_proj = self._project_features(features=self.features_encode)

    def inference_1step(self):
        x = self._word_embedding(inputs=tf.fill([tf.shape(self.features_encode)[0]], self._start))
        context_pre, self.alpha0 = self._attention_layer(self.features_encode, self.features_proj, self.init_h, reuse=False)
        context, beta = self._selector(context_pre, self.init_h, reuse=False)
        (self.c0, self.h0) = self._lstm(self.init_h, self.init_c, x, context, reuse=False)
        self.log_softmax0 = tf.nn.log_softmax(self._decode_lstm(x, self.h0, context, reuse=False))
        self.context4next0 = tf.reduce_sum(
            tf.reshape(self.resnet.features, [-1, self.L, tf.shape(self.resnet.features)[-1]]) * tf.expand_dims(self.alpha0, 2), 1)

    def inference_rest(self):
        x = self._word_embedding(inputs=self.in_word, reuse=True)
        context_pre, self.alpha = self._attention_layer(self.features_encode, self.features_proj, self.h_feed, reuse=True)
        context, beta = self._selector(context_pre, self.h_feed, reuse=True)
        (self.c, self.h) = self._lstm(self.h_feed, self.c_feed, x, context, reuse=True)
        self.log_softmax = tf.nn.log_softmax(self._decode_lstm(x, self.h, context, reuse=True))
        self.context4next = tf.reduce_sum(
            tf.reshape(self.resnet.features, [-1, self.L, tf.shape(self.resnet.features)[-1]]) * tf.expand_dims(self.alpha, 2), 1)

    def build_info_for2layer(self):
        self.embed4next = tf.reshape(self._word_embedding(inputs=self.word_feed, reuse=True), (-1,))

    # used for training
    def build_training(self):
        captions = self.captions
        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))
        captions_out -= 1 # this is crucial. The target caption includes <0> as null, while the output of the model doesn's include that.
        captions_out = tf.clip_by_value(captions_out, 0, 100000)
        alpha_list = []

        x = self._word_embedding(inputs=captions_in, reuse=True)
        local_loss = 0
        c = self.init_c
        h = self.init_h
        predict_list = []
        for t in range(16):
            context_pre, alpha = self._attention_layer(self.features_encode, self.features_proj, h, reuse=True)
            alpha_list.append(alpha)
            context, beta = self._selector(context_pre, h, reuse=True)

            (c, h) = self._lstm(h, c, x[:, t, :], context, reuse=True)
            logits = self._decode_lstm(x[:, t, :], h, context, reuse=True)
            predict_list.append(tf.nn.top_k(logits)[1])
            local_loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=captions_out[:, t]) * mask[:, t])
        self.loss += local_loss / tf.reduce_sum(mask)
        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
            alphas_all = tf.reduce_sum(alphas, 1)
            # todo: this is not the same with what we used in our model, but is better
            alpha_reg = self.alpha_c * tf.reduce_sum((16. / self.L - alphas_all) ** 2)
            self.loss += alpha_reg
        return self.loss