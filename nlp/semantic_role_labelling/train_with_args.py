import argparse
import util
import time
import model
import pickle
import numpy as np
import tensorflow as tf
import os
import sys


def preprocess_batch(a_batch):
    """
    Preprocess a batch and return input output sequences, actual lengths of each sequence in the batch
    and max length in the batch.
    :param a_batch:
    :return:
    """
    global FEATURE_LENGTH, VECTOR_EMBEDDINGS, NB_ARGUMENT_LABELS, USE_BI_LSTM
    _input_seqs = []
    _output_seqs = []
    _lengths = []
    _input_fw_seqs = []
    _input_bw_seqs = []
    for _ds in a_batch:
        _lengths.append(len(_ds.seq_features))
    _max_length = max(_lengths)

    for _ds in a_batch:
        _seq = []
        for feature in _ds.seq_features:
            element = []
            for l in feature:
                if isinstance(l, str):
                    element.extend(list(VECTOR_EMBEDDINGS.get(l, VECTOR_EMBEDDINGS["_unk_"])))
                else:
                    element.append(l)
            _seq.append(element)
        if USE_BI_LSTM:
            _input_fw_seqs.append(util.pad_sequence(_seq, _max_length, FEATURE_LENGTH))
            _input_bw_seqs.append(util.pad_sequence(list(reversed(_seq)), _max_length, FEATURE_LENGTH))
        else:
            _input_seqs.append(util.pad_sequence(_seq, _max_length, FEATURE_LENGTH))
        _output_seqs.append(util.pad_sequence(list(_ds.seq_labels), _max_length, 1))
    _input_seqs = np.array(_input_seqs)
    _input_fw_seqs = np.array(_input_fw_seqs)
    _input_bw_seqs = np.array(_input_bw_seqs)
    _output_seqs = np.array(_output_seqs)
    _lengths = np.array(_lengths)
    if USE_BI_LSTM:
        return _input_fw_seqs, _input_bw_seqs, _output_seqs, _lengths, _max_length
    else:
        return _input_seqs, _output_seqs, _lengths, _max_length


if __name__ == '__main__':
    print "======================================================================"
    print " ".join(sys.argv)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_win_size", help="predicate window size", default=1)
    parser.add_argument("--nb_epc", help="number of epochs", default=5)
    parser.add_argument("--model", help="model type", default="bi_lstm")
    parser.add_argument("--limit_size", help="limit on training samples", default=-1)
    parser.add_argument("--nb_lstm_layers", help="number of deeper lstm layers", default=2)
    parser.add_argument("--use_pos", help="whether to use POS information", default=0)
    parser.add_argument("--use_dep", help="whether to use dependency information", default=0)
    parser.add_argument("--reverse", help="whether to use reversed lstm", default=0)

    args = parser.parse_args()

    LIMIT_SIZE = int(args.limit_size)
    NB_EPOCHS = int(args.nb_epc)
    PRED_WINDOW_SIZE = int(args.pred_win_size)
    NB_DEEP_LSTM_LAYERS = int(args.nb_lstm_layers)
    MODEL_TO_USE = args.model
    MODEL_DICT = {"lstm": model.Model_span_based, "deep_lstm": model.Model_span_based_deeper_lstm,
                  "bi_lstm": model.Model_span_based_bi_lstm}
    USE_POS = bool(args.use_pos)
    USE_DEP = bool(args.use_dep)
    USE_BI_LSTM = MODEL_TO_USE == "bi_lstm"

    NB_ARGUMENT_LABELS = 54
    EMBEDDING_SIZE = 50

    FEATURE_LENGTH = 2 * EMBEDDING_SIZE + EMBEDDING_SIZE * (PRED_WINDOW_SIZE * 2 + 1) + 1 + int(USE_POS) + int(USE_DEP)
    start = time.time()
    TRAIN_SAMPLES = util.read_data(limit=LIMIT_SIZE, pred_win_size=PRED_WINDOW_SIZE, use_pos=USE_POS, use_dep=USE_DEP)
    VAL_SAMPLES = util.read_data("CoNLL2009-ST-English-development.txt", limit=LIMIT_SIZE,
                                 pred_win_size=PRED_WINDOW_SIZE,
                                 use_pos=USE_POS, use_dep=USE_DEP)
    print "Reading takes", time.time() - start
    try:
        with open('pretrained_vectors/needed' + '.pkl', 'rb') as f:
            VECTOR_EMBEDDINGS = pickle.load(f)
    except IOError:
        VECTOR_EMBEDDINGS = {"_unk_": np.zeros(shape=(EMBEDDING_SIZE,))}
    print "Vector embeddings loaded"

    tf.reset_default_graph()
    TRAIN_MODEL = MODEL_DICT[MODEL_TO_USE](feature_len=FEATURE_LENGTH, nb_layers=NB_DEEP_LSTM_LAYERS)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    SAVER = tf.train.Saver()
    print "Model loaded, start training ..."

    for nb_epc in range(NB_EPOCHS):
        print "Epoch %d:" % (nb_epc + 1)
        TRAIN_GEN = util.make_batch(TRAIN_SAMPLES)
        VAL_GEN = util.make_batch(VAL_SAMPLES)
        TRAIN_LOSS = []
        VAL_LOSS = []
        TRAIN_ACC = []
        VAL_ACC = []
        TRAIN_STATS = np.array([0, 0, 0, 0])
        VAL_STATS = np.array([0, 0, 0, 0])
        train_samples_processed = 0
        val_samples_processed = 0
        start = time.time()
        print "  training ..."
        while True:
            try:
                batch = TRAIN_GEN.next()
            except StopIteration:
                break
            else:
                if not USE_BI_LSTM:
                    input_seqs, output_seqs, lengths, max_length = preprocess_batch(batch)
                    train_samples_processed += input_seqs.shape[0]

                    _, train_loss, predictions = sess.run([TRAIN_MODEL.train_op, TRAIN_MODEL.loss,
                                                           TRAIN_MODEL.viterbi_sequence],
                                                          feed_dict={TRAIN_MODEL.input: input_seqs,
                                                                     TRAIN_MODEL.labels: output_seqs,
                                                                     TRAIN_MODEL.actual_lengths: lengths,
                                                                     TRAIN_MODEL.max_length_in_batch: max_length})
                else:
                    input_fw_seqs, input_bw_seqs, output_seqs, lengths, max_length = preprocess_batch(batch)
                    train_samples_processed += input_fw_seqs.shape[0]
                    _, train_loss, predictions = sess.run([TRAIN_MODEL.train_op, TRAIN_MODEL.loss,
                                                           TRAIN_MODEL.viterbi_sequence],
                                                          feed_dict={TRAIN_MODEL.input_fw: input_fw_seqs,
                                                                     TRAIN_MODEL.input_bw: input_bw_seqs,
                                                                     TRAIN_MODEL.labels: output_seqs,
                                                                     TRAIN_MODEL.actual_lengths: lengths,
                                                                     TRAIN_MODEL.max_length_in_batch: max_length})
                TRAIN_ACC.append(util.report_accuracy(lengths, max_length, output_seqs, predictions))
                TRAIN_LOSS.append(train_loss)
                TRAIN_STATS += util.report_TP_FP_FN(lengths, output_seqs, predictions)

        print "  evaluating ..."
        while True:
            try:
                batch = VAL_GEN.next()
            except StopIteration:
                break
            else:
                if not USE_BI_LSTM:
                    input_seqs, output_seqs, lengths, max_length = preprocess_batch(batch)
                    val_samples_processed += input_seqs.shape[0]
                    val_loss, predictions = sess.run([TRAIN_MODEL.loss, TRAIN_MODEL.viterbi_sequence],
                                                     feed_dict={TRAIN_MODEL.input: input_seqs,
                                                                TRAIN_MODEL.labels: output_seqs,
                                                                TRAIN_MODEL.actual_lengths: lengths,
                                                                TRAIN_MODEL.max_length_in_batch: max_length})
                else:
                    input_fw_seqs, input_bw_seqs, output_seqs, lengths, max_length = preprocess_batch(batch)
                    val_samples_processed += input_fw_seqs.shape[0]

                    val_loss, predictions = sess.run([TRAIN_MODEL.loss, TRAIN_MODEL.viterbi_sequence],
                                                     feed_dict={TRAIN_MODEL.input_fw: input_fw_seqs,
                                                                TRAIN_MODEL.input_bw: input_bw_seqs,
                                                                TRAIN_MODEL.labels: output_seqs,
                                                                TRAIN_MODEL.actual_lengths: lengths,
                                                                TRAIN_MODEL.max_length_in_batch: max_length})

                VAL_ACC.append(util.report_accuracy(lengths, max_length, output_seqs, predictions))
                VAL_LOSS.append(val_loss)
                VAL_STATS += util.report_TP_FP_FN(lengths, output_seqs, predictions)

        print "  done in %f:" % (time.time() - start)
        print "  train loss:", np.mean(TRAIN_LOSS)
        print "  val loss:", np.mean(VAL_LOSS)
        print "  train Precision/Recall/F1/Acc: %f %f %f" % util.report_precision_recall_f1(tuple(TRAIN_STATS)), \
            np.mean(TRAIN_ACC)
        print "  val Precision/Recall/F1/Acc: %f %f %f" % util.report_precision_recall_f1(tuple(VAL_STATS)), \
            np.mean(VAL_ACC)
        print "  val stats:", tuple(VAL_STATS)
        print "  %d/%d train/val samples processed" % (train_samples_processed, val_samples_processed)
    SAVER.save(sess, 'models/model-%s-%d-%d-%d.ckpt' % (MODEL_TO_USE, NB_DEEP_LSTM_LAYERS, USE_POS, USE_DEP))
