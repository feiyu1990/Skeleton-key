import tensorflow as tf
from model import HierarchicalModel
from beam_search import CaptionGenerator
import config
from utils import *
import h5py


def main():
    idx = h5py.File('./data/test/test_idx.h5')['labels']
    data = h5py.File('./data/test/test_caption.h5')
    model = HierarchicalModel(config, mode='inference')
    model.build()
    images = data['images']
    first_level_label_start_ix =  data['first_layer_label_start_ix']
    first_level_label_end_ix =  data['first_layer_label_end_ix']
    first_level_labels =  data['first_layer_labels']
    second_level_label_start_ix = data['label_start_ix']
    second_level_label_end_ix = data['label_end_ix']
    # second_level_label_pos = data['label_position']
    second_level_labels = data['labels']
    generator = CaptionGenerator(model, model.level1_word2ix, model.level2_word2ix,
                                 beam_size_1level=5, beam_size_2level=2,
                                 encourage_1level=0.1, encourage_2level=0.9)
    result = []
    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.per_process_gpu_memory_fraction=0.9
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
        tf.global_variables_initializer().run()
        for i in xrange(images.shape[0]):
            print '***************'
            images_batch = images[i:i+1, :, :, :]
            images_batch = crop_image(images_batch, False)
            prediction = generator.beam_search(sess, images_batch)
            print i, idx[i], prediction
            first_level_this = first_level_labels[first_level_label_start_ix[i]:first_level_label_end_ix[i]]
            second_level_this = []
            for j in xrange(first_level_label_start_ix[i], first_level_label_end_ix[i]):
                second_level_this.append(second_level_labels[second_level_label_start_ix[j]:second_level_label_end_ix[j]])
            decoded = decode_captions_2level(first_level_this, second_level_this, model.level1_model.idx_to_word, model.level2_model.idx_to_word)
            print decoded
            result.append({'image_id': int(idx[i]), 'caption': prediction})
    json.dump(result, open('./data/test/result_resnet_tf.json','w'))

if __name__ == "__main__":
    main()