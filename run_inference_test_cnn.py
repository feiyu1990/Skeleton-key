import tensorflow as tf
from ops import resnet
from beam_search import CaptionGenerator
import config
from utils import *
import h5py
import numpy as np


def main():
    img = h5py.File('data/testtest/testtest_caption.h5')['images'][:, :224, :224, :]
    img = np.asarray(img, dtype=np.float32)
    img /= 255
    model = resnet.ResNet()
    model.build_test()
    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.per_process_gpu_memory_fraction=0.9
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
        tf.global_variables_initializer().run()
        img_feature = sess.run(model.features, feed_dict={model.images: img})
        print img_feature
        print img_feature.shape
if __name__ == "__main__":
    main()