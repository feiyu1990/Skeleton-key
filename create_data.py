import json
import time
import string
import h5py
import argparse
from nltk import word_tokenize, pos_tag
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger
from random import shuffle, seed
import numpy as np
from scipy.misc import imread, imresize
import os

sent_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
data_root = './data/'
if not os.path.exists(data_root):
    os.mkdir(data_root)
for val_name in ['train', 'val', 'test']:
    if not os.path.exists(os.path.join(data_root, val_name)):
        os.mkdir(os.path.join(data_root, val_name))

def prepro(sent):
    temp = str(sent).lower().translate(None, string.punctuation).strip()
    try:
        int(temp.split(' ')[-1])
        temp = ' '.join(temp.split(' ')[:-1])
    except:
        pass
    token = word_tokenize(temp)
    return token, pos_tag(token)


def parse_dfs(sent_this, print_=False):
    visited = [(True, sent_this[0])]
    stem = []
    sent_flatten = sent_this[0].leaves()
    loc = []
    loc_pointer = len(sent_flatten)
    while len(visited) != 0:
        curr = visited[-1][1]
        curr_label = curr.label()
        record_this = visited[-1][0]
        visited.pop(-1)
        if curr.height() == 2:
            loc_pointer -= 1
            if record_this or loc_pointer == len(sent_flatten) - 1:
                stem.append(curr[0])
                loc.append(loc_pointer)
            continue
        for idx, i in enumerate(curr):
            record = True
            if i.height() <= 2:
                if curr_label == 'ADJP' and loc_pointer == len(sent_flatten):
                    # print 'HERE***', sent_flatten, curr
                    pass
                else:
                    if i.label()[0] in string.punctuation or (
                            curr_label in ('ADJP') and not (i[0] in ('next', 'full', 'ready'))
                            and not (idx == len(curr) - 1 and i.label() == 'NN')):
                        record = False
                    if curr_label == 'NP':
                        if not (
                                        (idx > 0 and curr[idx - 1].label().startswith('N') and i.label() == 'CC')
                                    or (i.label().startswith('N') and idx < len(curr) - 1 and curr[idx + 1].label() == 'CC')
                                or idx == len(curr) - 1
                            or (idx == len(curr) - 2 and curr[idx+1].label() == 'VBG')
                        ):
                            record=False
            visited.append((record, i))
    if print_:
        print sent_this[0].pretty_print()
        print stem[::-1], loc[::-1]
        print '++++++++++++++++++++++++++++++++++++++++++++'
    stem_pairs = []
    prev = 0
    for stem_i, i in enumerate(loc[::-1]):
        this_ = {sent_flatten[i]:sent_flatten[prev:i]}
        stem_pairs.append(this_)
        prev = i + 1
    return stem, stem_pairs


def refine_stem(stem_attr):
    new_stem_attr = []
    modify = False
    new_attr_to_add = None
    for temp in stem_attr:
        skeleton, attr = temp.items()[0]
        if new_attr_to_add:
            attr = new_attr_to_add + attr
            new_attr_to_add = None
        if len(attr) > 0 and attr != ['a'] and skeleton in ('grazing', 'flying', 'standing',
                        'plays', 'sits', 'stands',
                        'on', 'of', 'with', 'below', 'in', 'by', 'between', 'along', 'near',
                        'from', 'behind', 'above', 'at', 'down', 'while', 'around',
                        'stand', 'play', 'sit'
                        ):
                new_sk = attr[-1]
                new_attr = attr[:-1]
                new_stem_attr.append({new_sk: new_attr})
                new_stem_attr.append({skeleton: []})
                modify = True
        elif skeleton == 'looking' and len(attr) > 0:
            if len(new_stem_attr) > 0 and new_stem_attr[-1].keys()[0] == 'very':
                new_attr_to_add = new_stem_attr[-1].values()[0] + ['very'] + attr + ['looking']
                new_stem_attr = new_stem_attr[:-1]
            else:
                new_attr_to_add = attr + ['looking']
            modify = True
        elif skeleton == 'colored':
            if len(new_stem_attr) > 0:
                new_attr_to_add = new_stem_attr[-1].values()[0] + [new_stem_attr[-1].keys()[0]] + attr + [skeleton]
                new_stem_attr = new_stem_attr[:-1]
            else:
                new_attr_to_add = attr + [skeleton]
        else:
            new_stem_attr.append({skeleton: attr})
    new_stem = [i.keys()[0] for i in new_stem_attr]
    if modify:
        print stem_attr
        print '-------->'
        print new_stem_attr
        print new_stem

    return new_stem, new_stem_attr


def parsing_coco():
    info_raw = json.load(open(data_root + 'coco_raw.json'))
    caption_list = []
    for i in info_raw:
        caption_list.extend(i['captions'])
    print len(caption_list)

    end_idx = 5000
    caption_chunk = caption_list[:end_idx]
    caption_stem_all = []
    caption_attr_all = []
    while len(caption_chunk) > 0:
        t1 = time.time()
        token_pos = []
        for i in caption_chunk:
            token_pos.append(prepro(i))
        token, postag = zip(*token_pos)
        postag_stanford = st.tag_sents(token)
        tags = []
        for i,j in zip(postag, postag_stanford):
            temp1 = [1 if ii[1].startswith('N') else 0 for ii in i]
            temp2 = [1 if ii[1].startswith('N') else 0 for ii in j]
            if sum(temp1) > sum(temp2):
                tags.append(j)
            else:
                tags.append(i)

        parse = sent_parser.tagged_parse_sents(tags)
        caption_stem = []
        caption_attr = []
        for caption, sent in zip(caption_chunk, iter(parse)):
            # last_word = str(caption).translate(None, string.punctuation).strip().split(' ')[-1]
            sent_this = list(sent)
            stem, stem_attr_pair = parse_dfs(sent_this, print_=False)
            new_stem, new_stem_attr = refine_stem(stem_attr_pair)

            caption_stem.append(new_stem)
            caption_attr.append(new_stem_attr)
        t2 = time.time()
        print end_idx, 'processing time:', t2 - t1, caption_stem[-1]
        caption_stem_all.extend(caption_stem)
        caption_attr_all.extend(caption_attr)
        caption_chunk = caption_list[end_idx:end_idx+5000]
        end_idx += 5000

    all_info = []
    for j, k in zip(caption_stem_all, caption_attr_all):
        all_info.append((j, k))
    with open(data_root + 'caption_stem_attr_all.json', 'w') as f:
        json.dump(all_info, f)
    assert(len(caption_stem_all) == len(caption_list))


def combine_result():
    info_raw = json.load(open(data_root + 'coco_raw.json'))
    info_new = json.load(open(data_root + 'caption_stem_attr_all.json'))
    count = 0;with_loc = 0
    info_all = []
    for i in info_raw:
        attrs = []
        caption_stem = []
        for j in i['captions']:
            # temp = str(j).lower().translate(None, string.punctuation).strip().split()
            new_this = info_new[count]
            attr_this = new_this[1]
            caption_stem.append(' '.join(new_this[0]))
            attrs.append(attr_this)
            count += 1
        i['attr'] = attrs
        i['captions_stem'] = caption_stem
        info_all.append(i)
    print len(info_all), with_loc
    json.dump(info_all, open(data_root + 'coco_raw_with_attr.json', 'w'))


###################################
### then this is the to-h5 part ###
###################################

def prepro_captions(imgs, caption_name='captions'):
    # preprocess all the captions
    print 'example processed tokens:'
    for i, img in enumerate(imgs):
        img['processed_tokens'] = []
        for j, s in enumerate(img[caption_name]):
            txt = str(s).lower().translate(None, string.punctuation).strip().split()
            img['processed_tokens'].append(txt)
            if i < 10 and j == 0: print txt


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    print 'number of words in vocab would be %d' % (len(vocab),)
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print 'max length sentence in raw data: ', max_len
    print 'sentence length distribution (count, number of words):'
    sum_len = sum(sent_lengths.values())
    for i in xrange(max_len + 1):
        print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print 'inserting the special UNK token'
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


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
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
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


def encode_captions_attr(imgs, params, wtoi):
    imgs_flatten = [j for i in imgs for j in i['attr']]
    max_length = params['max_length']
    # N = len(imgs)
    N = sum(len(img['attr']) for img in imgs )  # total number of captions
    M = sum(len(sent) for img in imgs for sent in img['attr'] )  # total number of captions
    print M
    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    label_position = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, sent in enumerate(imgs_flatten):
        print sent
        n = len(sent)
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, temp in enumerate(sent):
            s = temp.values()[0]
            if len(s) > max_length:
                print s
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            label_position[caption_counter] = j+1
            caption_counter += 1
            if i % 100000 == 0:
                print s
            for k, w in enumerate(s):
                if k < max_length:
                    try:
                        Li[j, k] = wtoi[w]
                    except:
                        Li[j, k] = wtoi['UNK']

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    # assert np.all(label_length > 0), 'error: some caption had no words?'

    print 'encoded captions to array of size ', `L.shape`
    return L, label_position, label_start_ix, label_end_ix, label_length


def coco_h5(params):
    imgs = json.load(open(data_root + 'coco_raw_with_attr.json', 'r'))
    seed(123)  # make reproducible
    N = len(imgs)
    shuffle(imgs)  # shuffle the order
    dset = np.array((N, 256, 256, 3), dtype='uint8')
    for i,img in enumerate(imgs):
      I = imread(os.path.join(params['images_root'], img['file_path']))
      try:
          Ir = imresize(I, (256,256))
      except:
          print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
          raise
      if len(Ir.shape) == 2:
        Ir = Ir[:,:,np.newaxis]
        Ir = np.concatenate((Ir,Ir,Ir), axis=2)
      dset[i] = Ir
      if i % 1000 == 0:
        print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)

    idx_all = [i['id'] for i in imgs]

    #skeleton vocab
    prepro_captions(imgs, caption_name='captions_stem')
    vocab = build_vocab(imgs, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
    json.dump(itow, open(data_root + 'train/ix_to_word_stem.json', 'w'))
    json.dump(wtoi, open(data_root + 'train/word2ixs_stem.json', 'w'))

    #attribute vocab
    words_all = []
    for i in imgs:
        for j in i['attr']:
            for k in j:
                words_all.extend(k.values()[0])
    print words_all[:10]
    vocab = build_vocab(words_all, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    json.dump(itow, open(data_root + 'train/ix_to_word_attribute.json', 'w'))
    json.dump(wtoi, open(data_root + 'train/word2ixs_attr.json', 'w'))


    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)
    N = len(imgs)

    label_cut = label_start_ix[5000] - 1
    label_cut2 = label_start_ix[10000] - 1

    out = h5py.File(data_root + 'val/val_idx.h5', 'w')
    out.create_dataset('idx', data=idx_all[:5000], dtype='uint32')
    out.close()

    out = h5py.File(data_root + 'val/val_caption_stem.h5', 'w')

    out.create_dataset('label_end_ix', data=label_end_ix[:5000], dtype='uint32')
    out.create_dataset('label_start_ix', data=label_start_ix[:5000], dtype='uint32')
    out.create_dataset('label_length', data=label_length[:label_cut], dtype='uint32')
    out.create_dataset('labels', data=L[:label_cut, :], dtype='uint32')
    out.create_dataset('images', data=dset[:5000], dtype='uint8')
    out.close()

    out = h5py.File(data_root + 'test_caption_stem.h5', 'w')
    out.create_dataset('label_end_ix', data=label_end_ix[5000:10000] - label_cut, dtype='uint32')
    out.create_dataset('label_start_ix', data=label_start_ix[5000:10000] - label_cut, dtype='uint32')
    out.create_dataset('label_length', data=label_length[label_cut:label_cut2], dtype='uint32')
    out.create_dataset('labels', data=L[label_cut:label_cut2, :], dtype='uint32')
    out.create_dataset('images', data=dset[5000:10000], dtype='uint8')
    out.close()

    out = h5py.File(data_root + 'test_idx.h5', 'w')
    out.create_dataset('idx', data=idx_all[5000:10000], dtype='uint32')
    out.close()

    out = h5py.File(data_root + 'train/train_caption_stem.h5', "w")
    out.create_dataset('label_end_ix', data=label_end_ix[10000:] - label_cut2, dtype='uint32')
    out.create_dataset('label_start_ix', data=label_start_ix[10000:] - label_cut2, dtype='uint32')
    out.create_dataset('label_length', data=label_length[label_cut2:], dtype='uint32')
    out.create_dataset('labels', data=L[label_cut2:, :], dtype='uint32')
    out.create_dataset('images', data=dset[10000:], dtype='uint8')
    out.close()

    out = h5py.File(data_root + 'train/train_idx.h5', 'w')
    out.create_dataset('idx', data=idx_all[10000:], dtype='uint32')
    out.close()
    for val_name in ['train', 'test', 'val']:
        create_attribute_list_coco_tvt(val_name)


def create_attribute_list_coco_tvt(name):
    idx = h5py.File(data_root + name + '/' + name + '_idx.h5')['idx'][:]
    ix2w_stem = json.load(open(data_root + 'train/ix_to_word_stem.json'))
    w2ix_stem = dict([(j,int(i)) for i, j in ix2w_stem.items()])
    info_all = json.load(open(data_root + 'coco_raw_with_attr.json'))
    imgs = info_all
    imgs_new = []
    img_idx = []
    caption_real_count = 0
    for i in imgs:
        if i['id'] not in idx:
            continue
        img_idx.append(i['id'])
        i_new = {}
        for caption_stem_pre, caption_full_pre, caption_attr in zip(i['captions_stem'], i['captions'], i['attr']):
            caption_full = str(caption_full_pre).lower().translate(None, string.punctuation).strip().split()
            caption_stem = caption_stem_pre.lower().split(' ')
            labels_first_layer = [0] * 16
            for jj, ii in enumerate(caption_stem):
                if jj == 16:
                    break
                try:
                    labels_first_layer[jj] = w2ix_stem[ii]
                except:
                    labels_first_layer[jj] = w2ix_stem['UNK']
            caption_real_count += 1
            if len(i_new) == 0:
                i_new['file_path'] = i['file_path']
                i_new['id'] = i['id']
                i_new['captions'] = [caption_full_pre]
                i_new['first_layer_labels'] = [labels_first_layer]
                i_new['first_layer_label_length'] = [len(caption_stem)]
                i_new['first_layer_label_start_ix'] = caption_real_count
                i_new['attr'] = i['attr']
            else:
                i_new['captions'].append(caption_full_pre)
                i_new['first_layer_labels'].append(labels_first_layer)
                i_new['first_layer_label_length'].append(len(caption_stem))
        if caption_real_count % 10000 == 0:
            print i_new
        imgs_new.append(i_new)

    itow = json.load(open(data_root + 'train/ix_to_word_attribute.json'))
    wtoi = {w:int(i) for i,w in itow.items()}

    L, label_position, label_start_ix, label_end_ix, label_length = encode_captions_attr(imgs_new, params, wtoi)

    first_layer_labels = [i['first_layer_labels'] for i in imgs_new]
    print len(first_layer_labels)
    first_layer_labels = np.concatenate(first_layer_labels, axis=0)
    print first_layer_labels[:10]
    print first_layer_labels.shape


    first_layer_label_length = [i['first_layer_label_length'] for i in imgs_new]

    print len(first_layer_label_length)
    first_layer_label_length = np.concatenate(first_layer_label_length, axis=0)
    print first_layer_label_length[:10]
    print first_layer_label_length.shape

    first_layer_label_start_ix = [i['first_layer_label_start_ix'] for i in imgs_new]
    print first_layer_label_start_ix[:10]
    print first_layer_label_start_ix[-10:]


    first_layer_label_end_ix = [i-1 for i in first_layer_label_start_ix[1:]] + [len(first_layer_label_length)]
    print first_layer_label_end_ix[:10]
    print first_layer_label_end_ix[-10:]

    in_file = h5py.File(data_root + name + '/' + name + '_caption_stem.h5', 'r')

    f = h5py.File(data_root + name + '/' + name + '_caption.h5', "w")
    idx_new = [i['id']for i in imgs_new]
    assert((np.array(idx_new) == np.array(img_idx)).all())
    idx_dict = {}
    for dd, i in enumerate(list(idx)):
        idx_dict[i] = dd
    print(len(idx))
    print(len(idx_new))
    print(list(idx)[:10])
    print(idx_new[:10])
    img_before = in_file['images']
    img_new = np.zeros(img_before.shape, dtype='uint8')
    for i, j in enumerate(idx_new):
        img_new[i,:,:,:] = img_before[idx_dict[j], :, :, :]

    f.create_dataset("images", data=img_new, dtype='uint8')  # space for resized images
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("label_position", dtype='uint32', data=label_position)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    f.create_dataset("first_layer_label_length", dtype='uint32', data=first_layer_label_length)
    f.create_dataset("first_layer_labels", dtype='uint32', data=first_layer_labels)
    f.create_dataset("first_layer_label_start_ix", dtype='uint32', data=first_layer_label_start_ix)
    f.create_dataset("first_layer_label_end_ix", dtype='uint32', data=first_layer_label_end_ix)
    f.close()

    f = h5py.File(data_root + name + '/' + name + '_idx.h5', "w")
    f.create_dataset("labels", dtype='uint32', data=img_idx)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='/home/coco/images/',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    parsing_coco()
    combine_result()

    print 'parsed input parameters:'
    coco_h5(params)