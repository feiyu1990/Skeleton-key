from __future__ import division

import tensorflow as tf
import json
from ops import resnet, level1_model, level2_model


class HierarchicalModel(object):

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.images = None
        self.input_seqs = None
        # self.img_features = None
        # self.features_proj = None

        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')

        self.level1_word2ix = json.load(open('data/train/word2ix_stem.json'))
        self.level2_word2ix = json.load(open('data/train/word2ix_attr.json'))

        self.resnet = resnet.ResNet(images=self.images)
        self.level1_model = level1_model.Level1Model(word_to_idx=self.level1_word2ix,
                                                     dim_feature=config.LEVEL1_dim_feature,
                                                     dim_embed=config.LEVEL1_dim_embed,
                                                     dim_hidden=config.LEVEL1_dim_hidden,
                                                     alpha_c=config.LEVEL1_alpha, dropout=config.LEVEL1_dropout,
                                                     n_time_step=config.LEVEL1_T)

        self.level2_model = level2_model.Level2Model(word_to_idx=self.level2_word2ix,
                                                     dim_feature=config.LEVEL2_dim_feature,
                                                     dim_embed=config.LEVEL2_dim_embed,
                                                     dim_hidden=config.LEVEL2_dim_hidden,
                                                     dropout=config.LEVEL2_dropout, n_time_step=config.LEVEL2_T)
    def build(self):
        self.resnet.build_model(is_training=(self.mode == 'training'))
        self.level1_model.init_inference()
        self.level1_model.inference_1step()
        self.level1_model.inference_rest()
        self.level1_model.build_info_for2layer()
        self.level2_model.build_inference()