import json
import h5py
from random import shuffle, seed
import numpy as np
from scipy.misc import imread, imresize
import os
import argparse

def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            if len(s) == 0:
                print img
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print 'encoded captions to array of size ', `L.shape`
    return L, label_start_ix, label_end_ix, label_length


def coco_h5(params):
    root = '/home/hierarchical_lstm/level1_lstm/coco_dataset_1015/'
    imgs = json.load(open(root + 'coco_raw_with_attr.json', 'r'))
    seed(123)  # make reproducible
    shuffle(imgs)  # shuffle the order

    level1_word2ix = json.load(open('data/train/word2ix_stem.json'))
    level2_word2ix = json.load(open('data/train/word2ix_attr.json'))

    level1_label = np.zeros((len(imgs), 16), dtype='uint32')
    level2_label

    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, level1_word2ix)
    N = len(imgs)
    dset = np.zeros((N,3,256,256), dtype='uint8')
    for i, img in enumerate(imgs):
        I = imread(os.path.join(params['images_root'], img['file_path']))
        try:
            Ir = imresize(I, (256, 256))
        except:
            print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print 'processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N)

    # print imgs.shape

    idx_all = [i['id'] for i in imgs]

    label_cut = label_start_ix[5000] - 1
    label_cut2 = label_start_ix[10000] - 1


    out = h5py.File(root + 'val_idx_stem.h5', 'w')
    out.create_dataset('idx', data=idx_all[:5000], dtype='uint32')
    out.close()


    # h5_file = h5py.File(root + 'correct_train_caption.h5')
    out = h5py.File(root + 'val_caption_stem.h5', 'w')
    out.create_dataset('label_end_ix', data=label_end_ix[:5000], dtype='uint32')
    out.create_dataset('label_start_ix', data=label_start_ix[:5000], dtype='uint32')
    out.create_dataset('label_length', data=label_length[:label_cut], dtype='uint32')
    out.create_dataset('labels', data=L[:label_cut, :], dtype='uint32')
    out.create_dataset('images', data=dset[:5000, :, :, :], dtype='uint8')
    out.close()


    # h5_file = h5py.File(root + 'val_caption_1000.h5')
    out = h5py.File(root + 'test_caption_stem.h5', 'w')
    out.create_dataset('label_end_ix', data=label_end_ix[5000:10000] - label_cut, dtype='uint32')
    out.create_dataset('label_start_ix', data=label_start_ix[5000:10000] - label_cut, dtype='uint32')
    out.create_dataset('label_length', data=label_length[label_cut:label_cut2], dtype='uint32')
    out.create_dataset('labels', data=L[label_cut:label_cut2, :], dtype='uint32')
    out.create_dataset('images', data=dset[5000:10000, :, :, :], dtype='uint8')
    out.close()

    out = h5py.File(root + 'test_idx_stem.h5', 'w')
    out.create_dataset('idx', data=idx_all[5000:10000], dtype='uint32')
    out.close()

    # input = h5py.File(root + 'test_caption_stem.h5', 'r')
    out = h5py.File(root + 'train_caption_stem.h5', "w")
    out.create_dataset('label_end_ix', data=label_end_ix[10000:] - label_cut2, dtype='uint32')
    out.create_dataset('label_start_ix', data=label_start_ix[10000:] - label_cut2, dtype='uint32')
    out.create_dataset('label_length', data=label_length[label_cut2:], dtype='uint32')
    out.create_dataset('labels', data=L[label_cut2:, :], dtype='uint32')
    out.create_dataset('images', data=dset[10000:, :, :, :], dtype='uint8')
    out.close()

    out = h5py.File(root + 'train_idx_stem.h5', 'w')
    out.create_dataset('idx', data=idx_all[10000:], dtype='uint32')
    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='/home/yuwang/haystack/haystack2/project/data/bad_imgs/',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    # parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    #
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)

    coco_h5()