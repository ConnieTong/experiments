"""
Using tensorflow 1.4.0 (tensorflow 1.8 will not compile)
Forewords:
This script trains the Bi LSTM from models.py for 100 epochs on only semcor.data.xml+semcor.gold.key.bnids.txt. The
model produced achieves F1 score of 0.62 on ALL.data.xml+ALL.gold.key.bnids.txt. Because possible senses for each lemma
were not precisely defined, so it was assumed that all the possible senses for each lemma had been seen during
training (semcor+ALL). Therefore, please take into account that for some examples, this model may return the wrong
sense because of the limited amount of possible senses it had seen for that target lemma.
"""
from __future__ import print_function

import models
import tensorflow as tf
import read_data
import numpy as np
import time
import pickle
import random


def init_emb(vocab, pretrained_vec):
    """
    Instruction: see ==> set_up_embeddings.ipynb
    To init an embedding matrix from pretrained embeddings (the purpose of
    this function is not to load all the pretrained vectors but only the ones needed
    to save time and memory)
    This function is supposed to run only 1 time to init the word2vec and save it to disk.
    :param vocab: List of necessary words (make sure that this list covers both train and test set)
    :param pretrained_vec:
    :return: {word: vec}
    """
    word_vec = {"_unk_": np.random.uniform(-0.1, 0.1, 300)}
    for v in vocab:
        assert v not in word_vec
        if v in pretrained_vec:
            word_vec[v] = pretrained_vec[v]
        else:
            word_vec[v] = np.random.uniform(-0.1, 0.1, 300)

    # save the word2vec to disk so it can be loaded again next time
    with open('pretrained_vectors/needed' + '.pkl', 'wb') as _f:
        pickle.dump(word_vec, _f, pickle.HIGHEST_PROTOCOL)


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
        for _ds in _batch:
            _length_fw.append(len(_ds.fw))
            _length_bw.append(len(_ds.bw))
            _target = [1.0 if a == _ds.sense_id else 0.0 for a in _ds.possible_senses]
            for _ in range(MAX_NB_SENSES - len(_target)):
                _target.append(0.0)
            _targets.append(_target)
            _nb_senses.append(len(_ds.possible_senses))
            _target_ids.append(_ds.word_int)
        _max_length_fw = max(_length_fw)
        _max_length_bw = max(_length_bw)

        _output_fw_idx = []
        for _t in range(len(_batch)):
            _output_fw_idx.append([_t, _length_fw[_t]-1])

        _output_bw_idx = []
        for _t in range(len(_batch)):
            _output_bw_idx.append([_t, _length_bw[_t]-1])

        # if self.training:  # drop words
        #     for _ds in _batch:
        #         _ds.fw = [_t if random.random() > 0.1 else "_drop_" for _t in _ds.fw]
        #         _ds.bw = [_t if random.random() > 0.1 else "_drop_" for _t in _ds.bw]

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
               _target_ids, _nb_senses, _output_fw_idx, _output_bw_idx


if __name__ == "__main__":
    TRAIN_DATA, LEMMA2SENSES, LEMMA2INT = read_data.read_train_data(
        read_data.read_x("ALL.data.xml")[0], read_data.read_y("ALL.gold.key.bnids.txt"), True)
    MAX_NB_SENSES = max([len(LEMMA2SENSES[k]) for k in LEMMA2SENSES])  # max number of senses among all target words
    MAX_NB_TARGETS = len(LEMMA2SENSES)  # how many target words

    # load word embedding initialized by init_emb (run init_emb first if you don't have this file)
    with open('pretrained_vectors/needed' + '.pkl', 'rb') as f:
        WORD_VECTORS = pickle.load(f)
    WORD_VECTORS["_drop_"] = np.random.uniform(-0.1, 0.1, 300)  # add drop vector for drop words

    NB_EPOCHS = 100  # number of epochs to train
    x_val, y_val, _ = read_data.read_test_data(LEMMA2INT, LEMMA2SENSES, WORD_VECTORS)  # read validation data

    """train models"""
    tf.reset_default_graph()
    train_model = models.Model2(MAX_NB_SENSES, 32, MAX_NB_TARGETS)
    val_model = models.Model2(MAX_NB_SENSES, 32, MAX_NB_TARGETS, is_training=False)
    print("train models created")

    """run train models"""
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    VAL_ACC_LIST = []
    LOSS_LIST = []
    TRAIN_ACC_LIST = []
    start0 = time.time()
    saver = tf.train.Saver()
    for nb_epoch in range(NB_EPOCHS):
        TRAIN_TIME = 0
        READ_TIME = 0
        print("Running epoch %d" % (nb_epoch + 1))
        print("  trainable variables:", len(tf.trainable_variables()))
        start1 = time.time()
        TOTAL_LOSS = 0
        LIMIT = 0
        TRAIN_ACC = []
        TRAIN_BATCH_GEN = BatchGenerator(read_data.make_batch(TRAIN_DATA), is_training=True)

        # Train
        while True:
            try:
                start00 = time.time()
                _sample = TRAIN_BATCH_GEN.next()
                READ_TIME += time.time() - start00
            except StopIteration:
                break
            else:
                start00 = time.time()
                _, loss, train_acc = sess.run(
                    [train_model.train_op, train_model.loss, train_model.accuracy],
                    feed_dict={train_model.inputs_fw: _sample[0],
                               train_model.inputs_bw: _sample[1],
                               train_model.length_fw: _sample[2],
                               train_model.length_bw: _sample[3],
                               train_model.targets: _sample[4],
                               train_model.target_word_id: _sample[5],
                               train_model.nb_senses: _sample[6],
                               train_model.output_fw_index: _sample[7],
                               train_model.output_bw_index: _sample[8]})
                TOTAL_LOSS += loss
                LIMIT += 1
                TRAIN_TIME += time.time() - start00
                TRAIN_ACC.append(train_acc)

        print("  %d batches processed" % LIMIT)
        print("  loss is %f, train acc is %f" % (TOTAL_LOSS, np.mean(TRAIN_ACC)))
        LOSS_LIST.append(TOTAL_LOSS)
        TRAIN_ACC_LIST.append(np.mean(TRAIN_ACC))
        print("  epoch done in %f" % (time.time() - start1))

        # Validate
        VAL_BATCH_GEN = BatchGenerator(read_data.make_batch(x_val))
        TOTAL_ACC = []
        while True:
            try:
                batch = VAL_BATCH_GEN.next()
                acc = sess.run([val_model.accuracy],
                               feed_dict={val_model.inputs_fw: batch[0],
                                          val_model.inputs_bw: batch[1],
                                          val_model.length_fw: batch[2],
                                          val_model.length_bw: batch[3],
                                          val_model.targets: batch[4],
                                          val_model.target_word_id: batch[5],
                                          val_model.nb_senses: batch[6],
                                          val_model.output_fw_index: batch[7],
                                          val_model.output_bw_index: batch[8]})[0]
                TOTAL_ACC.append(acc)
            except StopIteration:
                break
        print("  val acc is %f" % np.mean(TOTAL_ACC))
        VAL_ACC_LIST.append(np.mean(TOTAL_ACC))
        print("  read takes %f, train takes %f" % (READ_TIME, TRAIN_TIME))
    print("training done in %f" % (time.time() - start0))

    # Save sess and some variables
    with open('losses2' + '.pkl', 'wb') as f:
        pickle.dump(LOSS_LIST, f, pickle.HIGHEST_PROTOCOL)
    with open('acc2' + '.pkl', 'wb') as f:
        pickle.dump(VAL_ACC_LIST, f, pickle.HIGHEST_PROTOCOL)
    with open('train_acc2' + '.pkl', 'wb') as f:
        pickle.dump(TRAIN_ACC_LIST, f, pickle.HIGHEST_PROTOCOL)
    saver.save(sess, '/mnt/video/nlp/model-main2.ckpt')
