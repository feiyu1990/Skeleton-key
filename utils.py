import numpy as np
import cPickle as pickle
# import hickle
import time
import os
import h5py
import json
import random

# def load_coco_data_dep(data_path='./data', split='train'):
#     data_path = os.path.join(data_path, split)
#     start_t = time.time()
#     data = {}
#
#     data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' %split))
#     with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
#         data['file_names'] = pickle.load(f)
#     with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
#         data['captions'] = pickle.load(f)
#     with open(os.path.join(data_path, '%s.image.idxs.pkl' %split), 'rb') as f:
#         data['image_idxs'] = pickle.load(f)
#
#     if split == 'train':
#         with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
#             data['word_to_idx'] = pickle.load(f)
#
#     for k, v in data.iteritems():
#         if type(v) == np.ndarray:
#             print k, type(v), v.shape, v.dtype
#         else:
#             print k, type(v), len(v)
#     end_t = time.time()
#     print "Elapse time: %.2f" %(end_t - start_t)
#     return data


def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
    f = h5py.File(os.path.join(data_path, '%s.data.h5' % split))
    data['features'] = f['feature']
    data['file_names'] = f['file_names']
    if 'img_idxs' in f:
        data['img_idxs'] = f['img_idxs']
    if 'captions' in f:
        data['captions'] = f['captions']

    # with open(os.path.join(data_path, '%s.file.names.pkl' % split), 'rb') as f:
    #     data['file_names'] = pickle.load(f)
    # with open(os.path.join(data_path, '%s.captions.pkl' % split), 'rb') as f:
    #     data['captions'] = pickle.load(f)
    # with open(os.path.join(data_path, '%s.image.idxs.pkl' % split), 'rb') as f:
    #     data['image_idxs'] = pickle.load(f)

    # if split == 'train':
    #     with open(os.path.join(data_path, 'word2ix.json'), 'rb') as f:
    #         data['word_to_idx'] = json.load(f)

    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" % (end_t - start_t)
    return data


def decode_helper(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape
    words_all = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == 'EOS':
                # words.append('<END>')
                break
            if word != 'NULL':
                words.append(word)
        words_all.append(words)
    return words_all


def decode_captions(captions, idx_to_word):
    return [' '.join(words) for words in decode_helper(captions, idx_to_word)]


def decode_captions_2level(level1_cap, level2_cap, level1_idx2word, level2_idx2word):
    first_levels = decode_helper(level1_cap, level1_idx2word)
    decodes = []
    for first_level, second_level in zip(first_levels, level2_cap):
        attrs_decoded = decode_helper(second_level, level2_idx2word)
        decode_this = []
        for i, first_word in enumerate(first_level):
            decode_this.extend(attrs_decoded[i])
            decode_this.append(first_word)
        decodes.append(' '.join(decode_this))
    return decodes


def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = []
    for i in mask:
        features.append(data['features'][i])
    features = np.asarray(features)
    # features = data['features'][mask]

    file_names = []
    for i in mask:
        file_names.append(data['file_names'][i])
    file_names = np.asarray(file_names)
    return features, file_names


def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
        f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
        f.write('METEOR: %f\n' %scores['METEOR'])  
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)


def crop_image(data, training):
    n_data = data.shape[0]
    # new_data = None
    if not training:
        # central crop
        start = (256-224)/2-1
        new_data = data[:, start:224+start, start:224+start, :]
    else:
        new_data = np.zeros((n_data, 224, 224, 3), dtype=np.float32)
        for i in xrange(n_data):
            start_x = random.randint(0, 256 - 224)
            start_y = random.randint(0, 256 - 224)
            new_data[i, :, :, :] = data[i, start_x:start_x+224, start_y:start_y+224, :]
    return new_data
