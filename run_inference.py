import tensorflow as tf
from model import HierarchicalModel
from beam_search import CaptionGenerator
import config
from utils import *

def main():
    model = HierarchicalModel(config, mode='inference')
    model.build()
    data = load_coco_data(data_path='./data', split='testtest')

    generator = CaptionGenerator(model, model.level1_word2ix, model.level2_word2ix,
                                 beam_size_1level=3, beam_size_2level=3,
                                 encourage_1level=1.0, encourage_2level=2.0)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # saver = tf.train.Saver()
        # saver.restore(sess, './model/pretrained_model-0')
        features_batch, image_files = sample_coco_minibatch(data, 1)
        generator.beam_search(sess, features_batch)
        # saver.save(sess, './model/pretrained_model', global_step=0)

if __name__ == "__main__":
    main()