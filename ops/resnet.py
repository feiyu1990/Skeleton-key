import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import h5py
import numpy as np

MOVING_AVERAGE_DECAY = 0.997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.00001

CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_RGB = [123.68, 116.779, 103.939]


model_load = h5py.File('./model/best_2level_cnn_param.h5')

class ResNet(object):
    def __init__(self):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')
        self.features = None

    def _get_variable(self, name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None

        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                               collections=collections, trainable=trainable)

    def _get_variable_const(self, name, initializer, weight_decay=0.0, dtype='float', trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None

        # todo: is it necessary to add to collection?
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

    def _bn(self, x, params_init, is_training):
        x_shape = x.get_shape()
        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable_const('beta', initializer=tf.constant(params_init['bias']))
        gamma = self._get_variable_const('gamma', initializer=tf.constant(params_init['weight']))
        moving_mean = self._get_variable_const('moving_mean',
                                               initializer=tf.constant(params_init['running_mean']), trainable=False)
        moving_variance = self._get_variable_const('moving_variance',
                                                   initializer=tf.constant(params_init['running_var']), trainable=False)
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        if ~is_training:
            mean = moving_mean
            variance = moving_variance

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        return x

    def _conv(self, x, params_init, stride):
        weights = self._get_variable_const('weights', initializer=tf.constant(params_init['weight']),
                                           weight_decay=CONV_WEIGHT_DECAY)
        bias = self._get_variable_const('bias', initializer=tf.constant(params_init['bias']))
        retval = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(retval, bias)

    def _pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _relu(self, x):
        return tf.nn.relu(x)

    def _bottleneck(self, x, n, is_training, block_name, bottleneck_name, stride):
        filters_in = x.get_shape()[-1]
        filters_out = 4 * n
        shortcut = x
        bottleneck_name = block_name + '/' + bottleneck_name
        print bottleneck_name
        with tf.variable_scope('a'):
            x = self._bn(x=x,
                         params_init={'bias': model_load[bottleneck_name + '/bn1_bias'][:],
                                      'weight': model_load[bottleneck_name + '/bn1_weight'][:],
                                      'running_mean': model_load[bottleneck_name + '/bn1_running_mean'][:],
                                      'running_var': model_load[bottleneck_name + '/bn1_running_var'][:]},
                         is_training=is_training)
            x = self._relu(x)
            x = self._conv(x=x,
                           params_init={'bias': model_load[bottleneck_name + '/conv3_bias'][:],
                                        'weight': np.transpose(model_load[bottleneck_name + '/conv3_weight'][:], (2, 3, 1, 0))},
                           stride=1)
        with tf.variable_scope('b'):
            x = self._bn(x=x,
                         params_init={'bias': model_load[bottleneck_name + '/bn4_bias'][:],
                                      'weight': model_load[bottleneck_name + '/bn4_weight'][:],
                                      'running_mean': model_load[bottleneck_name + '/bn4_running_mean'][:],
                                      'running_var': model_load[bottleneck_name + '/bn4_running_var'][:]},
                         is_training=is_training)
            x = self._relu(x)
            x = self._conv(x=x,
                           params_init={'bias': model_load[bottleneck_name + '/conv6_bias'][:],
                                        'weight': np.transpose(model_load[bottleneck_name + '/conv6_weight'][:], (2, 3, 1, 0))},
                           stride=stride)

        with tf.variable_scope('c'):
            x = self._bn(x=x,
                         params_init={'bias': model_load[bottleneck_name + '/bn7_bias'][:],
                                      'weight': model_load[bottleneck_name + '/bn7_weight'][:],
                                      'running_mean': model_load[bottleneck_name + '/bn7_running_mean'][:],
                                      'running_var': model_load[bottleneck_name + '/bn7_running_var'][:]},
                         is_training=is_training)
            x = self._relu(x)
            x = self._conv(x=x,
                           params_init={'bias': model_load[bottleneck_name + '/conv9_bias'][:],
                                        'weight': np.transpose(model_load[bottleneck_name + '/conv9_weight'][:], (2, 3, 1, 0))},
                           stride=1)
        assert(filters_out == x.get_shape()[-1])

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in:
                shortcut = self._conv(x=shortcut,
                                      params_init={'bias': model_load[block_name + '/shortcut1/conv1_bias'][:],
                                                   'weight': np.transpose(model_load[block_name + '/shortcut1/conv1_weight'][:], (2, 3, 1, 0))},
                                      stride=stride)

                shortcut = self._bn(x=shortcut,
                                    params_init={'bias': model_load[block_name + '/shortcut1/bn2_bias'][:],
                                                 'weight': model_load[block_name + '/shortcut1/bn2_weight'][:],
                                                 'running_mean': model_load[block_name + '/shortcut1/bn2_running_mean'][:],
                                                 'running_var': model_load[block_name + '/shortcut1/bn2_running_var'][:]},
                                    is_training=is_training)
        assert(filters_out == shortcut.get_shape()[-1])

        return x + shortcut

    def build_model(self, is_training):
        h = (self.images - np.array(IMAGENET_MEAN_RGB))
        h = tf.divide(h, 255)
        self.testtest = h
        with tf.variable_scope('resnet/block1'):
            h = self._conv(x=h,
                           params_init={'bias': model_load['cnn1/bias'][:],
                                        'weight': np.transpose(model_load['cnn1/weight'][:], (2, 3, 1, 0))},
                           stride=2)
            h = self._bn(x=h,
                         params_init={'bias': model_load['bn1/bias'][:],
                                      'weight': model_load['bn1/weight'][:],
                                      'running_mean': model_load['bn1/running_mean'][:],
                                      'running_var': model_load['bn1/running_var'][:]},
                         is_training=is_training)
            h = self._relu(h)
            h = self._pool(h)

        # block5
        with tf.variable_scope('resnet/block5'):
            for i in xrange(1, 4):
                with tf.variable_scope('bottleneck' + str(i)):
                    h = self._bottleneck(h, 64, is_training, 'block5', 'bottleneck' + str(i), 1)

        self.img_result = h

        # block6
        with tf.variable_scope('resnet/block6'):
            with tf.variable_scope('bottleneck1'):
                h = self._bottleneck(h, 128, is_training, 'block6', 'bottleneck1', 2)
            for i in xrange(2, 25):
                with tf.variable_scope('bottleneck' + str(i)):
                    h = self._bottleneck(h, 128, is_training, 'block6', 'bottleneck' + str(i), 1)

        # block7
        with tf.variable_scope('resnet/block7'):
            with tf.variable_scope('bottleneck1'):
                h = self._bottleneck(h, 256, is_training, 'block7', 'bottleneck1', 2)
            for i in xrange(2, 37):
                with tf.variable_scope('bottleneck' + str(i)):
                    h = self._bottleneck(h, 256, is_training, 'block7', 'bottleneck' + str(i), 1)

        # block8
        with tf.variable_scope('resnet/block8'):
            with tf.variable_scope('bottleneck1'):
                h = self._bottleneck(h, 512, is_training, 'block8', 'bottleneck1', 2)
            for i in xrange(2, 4):
                with tf.variable_scope('bottleneck' + str(i)):
                    h = self._bottleneck(h, 512, is_training, 'block8', 'bottleneck' + str(i), 1)

        with tf.variable_scope('resnet/block10'):
            h = self._bn(x=h,
                         params_init={'bias': model_load['bn10/bias'][:],
                                      'weight': model_load['bn10/weight'][:],
                                      'running_mean': model_load['bn10/running_mean'][:],
                                      'running_var': model_load['bn10/running_var'][:]},
                         is_training=is_training)
            h = self._relu(h)
        self.features = h





