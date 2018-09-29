import numpy as np
import tensorflow as tf
import util
import sys
import model
import pickle
import collections
from sklearn.metrics import confusion_matrix


def preprocess_batch(a_batch):
    global FEATURE_LENGTH, VECTOR_EMBEDDINGS, NB_ARGUMENT_LABELS
    _input_seqs = []
    _output_seqs = []
    _lengths = []
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

        _input_seqs.append(util.pad_sequence(_seq, _max_length, FEATURE_LENGTH))
        _output_seqs.append(util.pad_sequence(list(_ds.seq_labels), _max_length, 1))
    _input_seqs = np.array(_input_seqs)
    _output_seqs = np.array(_output_seqs)
    _lengths = np.array(_lengths)
    return _input_seqs, _output_seqs, _lengths, _max_length


def main(write_pred=True, limit=None):
    global FEATURE_LENGTH, VECTOR_EMBEDDINGS, NB_ARGUMENT_LABELS, EMBEDDING_SIZE, PRED_WINDOW_SIZE
    EMBEDDING_SIZE = 50
    PRED_WINDOW_SIZE = 3
    FEATURE_LENGTH = 2 * EMBEDDING_SIZE + EMBEDDING_SIZE * (PRED_WINDOW_SIZE * 2 + 1) + 3
    NB_ARGUMENT_LABELS = 54
    test_file = "CoNLL2009-ST-English-development.txt"
    try:
        with open('pretrained_vectors/needed' + '.pkl', 'rb') as f:
            VECTOR_EMBEDDINGS = pickle.load(f)
    except IOError:
        VECTOR_EMBEDDINGS = {"_unk_": np.zeros(shape=(EMBEDDING_SIZE,))}
    print "Vector embeddings loaded"

    val_samples = util.read_data_by_packet(file_name=test_file, limit=-1,
                                           pred_win_size=PRED_WINDOW_SIZE)
    if limit is None:
        limit = len(val_samples)
    keys = sorted(val_samples.keys())[:limit]

    tf.reset_default_graph()
    train_model = model.Model_span_based_deeper_lstm(feature_len=FEATURE_LENGTH, nb_layers=2)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    if write_pred:
        sys.stdout = open("demo_pred.txt", "w")
    else:
        sys.stdout = open("demo_gold.txt", "w")
    true_labels = []  # for confusion matrix
    pred_labels = []  # for confusion matrix
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, 'models/model-deep_lstm-2-1-1.ckpt')
        val_samples_processed = 0
        for k in keys:
            to_write, to_predict = val_samples[k]
            predictions_matrix = []
            batch_gen = util.make_batch(to_predict)
            while True:
                try:
                    batch = batch_gen.next()
                except StopIteration:
                    break
                else:
                    input_seqs, output_seqs, lengths, max_length = preprocess_batch(batch)
                    val_samples_processed += input_seqs.shape[0]

                    pred = sess.run([train_model.viterbi_sequence],
                                    feed_dict={train_model.input: input_seqs,
                                               train_model.actual_lengths: lengths,
                                               train_model.max_length_in_batch: max_length})[0]
                    for l in range(pred.shape[0]):
                        predictions_matrix.append(map(lambda x: util.INV_CLASSES[x], pred[l]))
                        true_labels.extend(map(lambda x: util.INV_CLASSES[x], output_seqs[l, :lengths[l]]))
                        pred_labels.extend(map(lambda x: util.INV_CLASSES[x], pred[l, :lengths[l]]))

            if not write_pred:
                predictions_matrix = []
                for ds in to_predict:
                    pred = map(lambda x: util.INV_CLASSES[x], ds.seq_labels)
                    predictions_matrix.append(pred)
            predictions_matrix = np.array(predictions_matrix)

            if predictions_matrix.shape[0] > 0:
                predictions_matrix = np.transpose(predictions_matrix)
                result = np.concatenate((to_write, predictions_matrix), axis=1)
            else:
                result = to_write

            for g in range(result.shape[0]):
                print "\t".join(list(result[g]))
            print ""
    sys.stdout = sys.__stdout__
    print ""
    if "test" not in test_file:
        print "computing confusion matrix"
        mat = confusion_matrix(true_labels, pred_labels, labels=['_', 'A1', 'A0', 'A2', 'AM-TMP', 'AM-MNR',
                                                                 'AM-LOC', 'AM-MOD', 'A3', 'AM-ADV'])
        print mat.shape
        np.save("conf_mat.npy", mat)
        print collections.Counter(true_labels)


if __name__ == "__main__":
    main(limit=None)
    # gold = main(limit=1)
    # pred = main(write_pred=False, limit=1)
    # print [gold[t].shape[0] for t in range(gold.shape[0])]
    # print util.report_TP_FP_FN([gold[t].shape[0] for t in range(gold.shape[0])], gold, pred)

