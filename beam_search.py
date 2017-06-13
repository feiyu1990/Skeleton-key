# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class for generating captions from an image-to-text model."""

from __future__ import absolute_import
from __future__ import division

import heapq

import numpy as np
from utils import *

class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, c, h, logprob, score, embeds=None, contexts=None, hiddens=None, info=True):
        """Initializes the Caption.
        Args:
          sentence: List of word ids in the caption.
          logprob: Log-probability of the caption.
          score: Score of the caption.
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.c = c
        self.h = h
        self.logprob = logprob
        self.score = score
        if info:
            self.embeds = embeds
            self.contexts = contexts
            self.hiddens = hiddens

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self,
                 model,
                 vocab_1level, vocab_2level,
                 beam_size_1level=3, beam_size_2level=3,
                 max_caption_length_1level=16, max_caption_length_2level=6,
                 length_normalization_factor=0.0,
                 encourage_1level=0.0, encourage_2level=0.0,
                 level2=True):
        """Initializes the generator.
        Args:
          model: Object encapsulating a trained image-to-text model. Must have
            methods feed_image() and inference_step(). For example, an instance of
            InferenceWrapperBase.
          vocab: A Vocabulary object.
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor: If != 0, a number x such that captions are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of captions depending on their lengths. For example, if
            x > 0 then longer captions will be favored.
        """
        self.vocab_1level = vocab_1level
        self.vocab_2level = vocab_2level
        self.model = model

        self.beam_size_1level = beam_size_1level
        self.beam_size_2level = beam_size_2level
        self.max_caption_length_1level = max_caption_length_1level
        self.max_caption_length_2level = max_caption_length_2level
        self.length_normalization_factor = length_normalization_factor
        self.encourage_1level = encourage_1level
        self.encourage_2level = encourage_2level
        self.level2 = level2

    def beam_search(self, sess, img):
        """Runs beam search caption generation on a single image.
        Args:
          sess: TensorFlow Session object.
          encoded_image: An encoded image string.
        Returns:
          A list of Caption sorted by descending score.
        """
        resnet = self.model.resnet
        level1 = self.model.level1_model
        level2 = self.model.level2_model

        # feed image into resnet and get image features
        img_feature = sess.run(resnet.features, feed_dict={self.model.images: img})

        # level1 (skeleton)
        (init_c, init_h, features_encode, features_proj) = sess.run(
            [level1.init_c, level1.init_h, level1.features_encode, level1.features_proj],
            feed_dict = {level1.features: img_feature})

        initial_beam = Caption(
            sentence=[self.vocab_1level['START']],
            c=init_c,
            h=init_h,
            logprob=0.0,
            score=0.0, embeds=[], contexts=[], hiddens=[])
        partial_captions = TopN(self.beam_size_1level)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size_1level)

        # Run beam search.
        for t in range(self.max_caption_length_1level):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            h_feed = np.reshape(np.array([c.h for c in partial_captions_list]), (-1, level1.H))
            c_feed = np.reshape(np.array([c.c for c in partial_captions_list]), (-1, level1.H))
            if t == 0:
                (c, h, log_softmax, alpha, context) = sess.run([level1.c0, level1.h0, level1.log_softmax0, level1.alpha0, level1.context4next0],
                                               feed_dict={level1.features: img_feature})
            else:
                (c, h, log_softmax, alpha, context) = sess.run([level1.c, level1.h, level1.log_softmax, level1.alpha, level1.context4next],
                                                      feed_dict={level1.c_feed: c_feed, level1.h_feed: h_feed,
                                                                 level1.in_word: input_feed, level1.features: img_feature})

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = log_softmax[i]
                word_probabilities[:level1._eos - 1] += self.encourage_1level
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities, start=1))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size_1level]
                # Each next word gives a new partial caption.
                for w, logp in words_and_probs:
                    embed = sess.run(level1.embed4next, feed_dict={level1.word_feed: np.array([w])})
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + logp
                    score = logprob
                    if w == level1.word_to_idx['EOS']:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, c[i], h[i], logprob, score,
                                       partial_caption.embeds, partial_caption.contexts, partial_caption.hiddens)

                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, c[i], h[i], logprob, score,
                                       partial_caption.embeds + [embed],
                                       partial_caption.contexts + [context[i]],
                                       partial_caption.hiddens + [h[i]])
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if not complete_captions.size():
            complete_captions = partial_captions

        level1_top_captions = complete_captions.extract(sort=True)

        full_sentence = []

        if self.level2:
            # level2 (attributes)
            for caption in level1_top_captions:
                sentence_level1 = caption.sentence[1:]
                embeds, contexts, hiddens = caption.embeds, caption.contexts, caption.hiddens
                # print sentence_level1
                # print decode_captions(np.squeeze(np.asarray(sentence_level1)), level1.idx_to_word)
                words_level1 = decode_captions(np.squeeze(np.asarray(sentence_level1)), level1.idx_to_word)[0].split(' ')
                attrs_level2 = []
                for t_level1 in xrange(len(embeds)):
                    embed = np.reshape(embeds[t_level1], (1, -1))
                    context = np.reshape(contexts[t_level1], (1, -1))
                    hidden = np.reshape(hiddens[t_level1], (1, -1))
                    (init_c, init_h) = sess.run([level2.init_c, level2.init_h],
                                                feed_dict={level2.embedding: embed, level2.context: context, level2.hidden: hidden})

                    initial_beam = Caption(
                                sentence=[self.vocab_2level['START']],
                                c=init_c,
                                h=init_h,
                                logprob=0.0,
                                score=0.0, info=False)
                    partial_captions = TopN(self.beam_size_2level)
                    partial_captions.push(initial_beam)
                    complete_captions = TopN(self.beam_size_2level)

                    # Run beam search.
                    for t in range(self.max_caption_length_2level):
                        partial_captions_list = partial_captions.extract()
                        partial_captions.reset()
                        input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
                        h_feed = np.reshape(np.array([c.h for c in partial_captions_list]), (-1, level2.H))
                        c_feed = np.reshape(np.array([c.c for c in partial_captions_list]), (-1, level2.H))
                        (c, h, log_softmax) = sess.run([level2.c, level2.h, level2.log_softmax],
                                                       feed_dict={level2.c_feed: c_feed, level2.h_feed: h_feed,
                                                                  level2.in_word: input_feed})

                        for i, partial_caption in enumerate(partial_captions_list):
                            word_probabilities = log_softmax[i]
                            word_probabilities[:level2._eos - 1] += self.encourage_2level
                            words_and_probs = list(enumerate(word_probabilities, start=1))
                            words_and_probs.sort(key=lambda x: -x[1])
                            words_and_probs = words_and_probs[0:self.beam_size_2level]
                            for w, logp in words_and_probs:
                                sentence = partial_caption.sentence + [w]
                                logprob = partial_caption.logprob + logp
                                score = logprob
                                if w == level2.word_to_idx['EOS']:
                                    if self.length_normalization_factor > 0:
                                        score /= len(sentence) ** self.length_normalization_factor
                                    beam = Caption(sentence, c[i], h[i], logprob, score, info=False)
                                    complete_captions.push(beam)
                                else:
                                    beam = Caption(sentence, c[i], h[i], logprob, score, info=False)
                                    partial_captions.push(beam)
                        if partial_captions.size() == 0:
                            # We have run out of partial candidates; happens when beam_size = 1.
                            break
                    if not complete_captions.size():
                        complete_captions = partial_captions
                    attr = decode_captions(np.squeeze(np.asarray(complete_captions.extract(sort=True)[0].sentence))[1:], level2.idx_to_word)
                    attrs_level2.extend(attr)
                # print words_level1, attrs_level2

                print ' '.join([j if i == '' else i + ' ' + j for (j, i) in zip(words_level1, attrs_level2)])
                full_sentence.append(' '.join([i + ' ' + j for (j, i) in zip(words_level1, attrs_level2)]))
        else:
            full_sentence = [i[1:] for i in level1_top_captions]
            print full_sentence
        return full_sentence[0]