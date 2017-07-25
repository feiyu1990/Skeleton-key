from __future__ import division

import tensorflow as tf
import h5py
import numpy as np



class Level2Model(object):
    def __init__(self, word_to_idx, dim_feature = 2048, dim_prev_emb = 512, dim_prev_cont = 2048, dim_prev_hidd = 1800,
                 dim_embed=512, dim_hidden=1024, n_time_step=6, dropout=True):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.D = dim_feature
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['START']
        self._null = word_to_idx['NULL']
        self._eos = word_to_idx['EOS']

        # Place holder for features and captions
        self.embedding = tf.placeholder(tf.float32, [None, dim_prev_emb])
        self.context = tf.placeholder(tf.float32, [None, dim_prev_cont])
        self.hidden = tf.placeholder(tf.float32, [None, dim_prev_hidd])
        self.model_load = h5py.File('model/best_2level_attrlm_param.h5')

        self.init_c = None
        self.init_h = None
        self.c_feed = None
        self.h_feed = None
        self.in_word = None
        self.c = None
        self.h = None
        self.log_softmax = None

        w_numpy_pre = self.model_load['/lookup_table/weight'][:]
        w_numpy = np.zeros((w_numpy_pre.shape[0] + 1, w_numpy_pre.shape[1]))
        w_numpy[1:, :] = w_numpy_pre
        self.w_numpy = w_numpy.astype(np.float32)

    def _cnn_encoding(self, embedding, context, hidden):
        with tf.variable_scope('level2_cnn_encoding'):
            weights_w = np.transpose(self.model_load['/cnn_encoding/weight_w2i'][:], (1, 0))
            weights_i = np.transpose(self.model_load['/cnn_encoding/weight_i2i'][:], (1, 0))
            weights_r = np.transpose(self.model_load['/cnn_encoding/weight_r2i'][:], (1, 0))
            bias_w = self.model_load['/cnn_encoding/bias_w2i'][:]
            bias_i = self.model_load['/cnn_encoding/bias_i2i'][:]
            bias_r = self.model_load['/cnn_encoding/bias_r2i'][:]
            out = tf.matmul(embedding, weights_w) + bias_w
            out += tf.matmul(context, weights_i) + bias_i
            out += tf.matmul(hidden, weights_r) + bias_r
            out = tf.nn.relu(out)
            return out

    def _word_embedding(self, inputs, reuse=False):
        # todo: word_embedding for <START> token is all-zero
        with tf.variable_scope('level2_word_embedding', reuse=reuse):
            w = tf.get_variable('w', initializer=self.w_numpy)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')
            return x

    def _lstm(self, input_h, input_c, input_x, reuse=False):
        with tf.variable_scope('level2_lstm', reuse=reuse):
            w_i2h_ = np.transpose(self.model_load['/core/i2h_1/weight'][:], (1, 0))
            b_i2h_ = self.model_load['/core/i2h_1/bias'][:]
            w_h2h_ = np.transpose(self.model_load['/core/h2h_1/weight'][:], (1, 0))
            b_h2h_ = self.model_load['/core/h2h_1/bias'][:]

            w_i2h = tf.get_variable('w_i2h', initializer=w_i2h_)
            b_i2h = tf.get_variable('b_i2h', initializer=b_i2h_)
            w_h2h = tf.get_variable('w_h2h', initializer=w_h2h_)
            b_h2h = tf.get_variable('b_h2h', initializer=b_h2h_)

            input_x = tf.cast(input_x, tf.float32)
            i2h = tf.matmul(input_x, w_i2h) + b_i2h
            h2h = tf.matmul(input_h, w_h2h) + b_h2h
            all_input_sums = i2h + h2h
            reshaped = tf.reshape(all_input_sums, [-1, 4, self.H])
            n1, n2, n3, n4 = tf.unstack(reshaped, axis=1)
            in_gate = tf.sigmoid(n1)
            forget_gate = tf.sigmoid(n2)
            out_gate = tf.sigmoid(n3)
            in_transform = tf.tanh(n4)
            c = tf.multiply(forget_gate, input_c) + tf.multiply(in_gate, in_transform)
            h = tf.multiply(out_gate, tf.tanh(c))
            return c, h

    def _decode_lstm(self, h, dropout=False, reuse=False):
        with tf.variable_scope('level2_logits', reuse=reuse):
            w_out_ = np.transpose(self.model_load['/core/decoder/weight'][:], (1, 0))
            b_out_ = self.model_load['/core/decoder/bias'][:]
            w_out = tf.get_variable('w_out', initializer=w_out_)
            b_out = tf.get_variable('b_out', initializer=b_out_)

            # if dropout:
                # h --> top_h
                # h = tf.nn.dropout(h, 0.5)

            # h_logits --> h_out
            out_logits = tf.matmul(h, w_out) + b_out
            out_logsoftmax = tf.nn.log_softmax(out_logits)
            return out_logsoftmax

    def build_inference(self):
        embed = self.embedding
        context = self.context
        hidden = self.hidden
        features = self._cnn_encoding(embedding=embed, context=context, hidden=hidden)
        c = tf.zeros([tf.shape(self.context)[0], self.H])
        h = tf.zeros([tf.shape(self.context)[0], self.H])
        (self.init_c, self.init_h) = self._lstm(h, c, features, reuse=False)
        _ = self._decode_lstm(self.init_h)
        _ = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))

        self.in_word = tf.placeholder(tf.int32, [None])
        x = self._word_embedding(inputs=self.in_word, reuse=True)

        self.c_feed = tf.placeholder(tf.float32, [None, self.H])
        self.h_feed = tf.placeholder(tf.float32, [None, self.H])
        (self.c, self.h) = self._lstm(self.h_feed, self.c_feed, x, reuse=True)
        self.log_softmax = self._decode_lstm(self.h, reuse=True)
