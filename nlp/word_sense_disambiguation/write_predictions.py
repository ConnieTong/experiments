from __future__ import print_function

import read_data
import models
import pickle
import tensorflow as tf
import numpy as np
import sys


def init_vector(word):
    """
    Return a pretrained vector for a given word, return unk vector
    if the word is not in pretrained vectors
    :param word:
    :return: vector
    """
    global WORD_VECTORS
    return WORD_VECTORS.get(word, WORD_VECTORS["_unk_"])


def pad_sequence(seq, max_length):
    """
    Pad a sequence with zeros till the max length
    :param seq: [w1, w2, w3 ...]
    :param max_length:
    :return:
    """
    for _ in range(max_length - len(seq)):
        seq.append(np.zeros(300))
    return seq


class BatchGenerator:
    """
    Custom generator for threading
    """

    def __init__(self, it, is_training=False):
        """
        :param it: a regular generator
        :param is_training: whether to drop words during training
        """
        self.it = it
        self.training = is_training

    def __iter__(self):
        return self

    def next(self):
        _batch = self.it.next()
        _input_fw = []
        _input_bw = []
        _length_fw = []
        _length_bw = []
        _targets = []
        _nb_senses = []
        _target_ids = []
        _sample_ids = []

        for _ds in _batch:
            _length_fw.append(len(_ds.fw))
            _length_bw.append(len(_ds.bw))
            _target = [1.0 if a == _ds.sense_id else 0.0 for a in _ds.possible_senses]
            for _ in range(MAX_NB_SENSES - len(_target)):
                _target.append(0.0)
            _targets.append(_target)
            _nb_senses.append(len(_ds.possible_senses))
            _target_ids.append(_ds.word_int)
            _sample_ids.append(_ds.word_id)

        _max_length_fw = max(_length_fw)
        _max_length_bw = max(_length_bw)

        _output_fw_idx = []
        for _t in range(len(_batch)):
            _output_fw_idx.append([_t, _length_fw[_t] - 1])

        _output_bw_idx = []
        for _t in range(len(_batch)):
            _output_bw_idx.append([_t, _length_bw[_t] - 1])

        for _ds in _batch:
            _input_fw.append(pad_sequence([init_vector(b) for b in _ds.fw], _max_length_fw))
            _input_bw.append(pad_sequence(list(reversed([init_vector(c) for c in _ds.bw])), _max_length_bw))
        _input_fw = np.array(_input_fw)
        _input_bw = np.array(_input_bw)
        _length_fw = np.array(_length_fw)
        _length_bw = np.array(_length_bw)
        _targets = np.array(_targets)
        _target_ids = np.array(_target_ids)
        _nb_senses = np.array(_nb_senses)
        return _input_fw, _input_bw, _length_fw, _length_bw, _targets, \
               _target_ids, _nb_senses, _output_fw_idx, _output_bw_idx, _sample_ids


TRAIN_DATA, LEMMA2SENSES, LEMMA2INT = read_data.read_train_data(
    read_data.read_x("ALL.data.xml")[0], read_data.read_y("ALL.gold.key.bnids.txt"), True)
MAX_NB_SENSES = max([len(LEMMA2SENSES[k]) for k in LEMMA2SENSES])  # max number of senses among all target words
MAX_NB_TARGETS = len(LEMMA2SENSES)  # how many target words

# load word embedding initialized by init_emb (run init_emb first if you don't have this file)
with open('pretrained_vectors/needed' + '.pkl', 'rb') as f:
    WORD_VECTORS = pickle.load(f)

TEST_DATA = read_data.read_test_words(LEMMA2INT, LEMMA2SENSES)
VALID_TEST_DATA = [point[0] for point in TEST_DATA if point[1] == "in"]
print("%d/%d are valid" % (len(VALID_TEST_DATA), len(TEST_DATA)))

tf.reset_default_graph()
val_model = models.Model(MAX_NB_SENSES, 32, MAX_NB_TARGETS, is_training=False)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
TEST_GEN = BatchGenerator(read_data.make_batch(VALID_TEST_DATA))
res = {}
with tf.Session() as sess:
    sess.run(init)
    print('Loading Model...')
    saver.restore(sess, '/mnt/video/nlp/model-main2.ckpt')
    while True:
        try:
            batch = TEST_GEN.next()
            pred = sess.run([val_model.prediction],
                            feed_dict={val_model.inputs_fw: batch[0],
                                       val_model.inputs_bw: batch[1],
                                       val_model.length_fw: batch[2],
                                       val_model.length_bw: batch[3],
                                       val_model.targets: batch[4],
                                       val_model.target_word_id: batch[5],
                                       val_model.nb_senses: batch[6],
                                       val_model.output_fw_index: batch[7],
                                       val_model.output_bw_index: batch[8]})[0]
            for i in range(len(batch[9])):
                res[batch[9][i]] = pred[i]
        except StopIteration:
            break

# Write predictions
sys.stdout = open("test_answer.txt", "w")
for test_sample in TEST_DATA:
    if test_sample[1] == "in":
        print("%s\t%s\n" % (test_sample[0].word_id, LEMMA2SENSES[test_sample[2]][res[test_sample[0].word_id]]))
    else:
        print("%s\t%s\n" % (test_sample[0], "unk"))