import tf_lstm_model
import util
import tensorflow as tf
import numpy as np
import time
import os
import argparse
import sys
import pickle

if __name__ == '__main__':
    print("")
    print("======================================================================")
    print(" ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", help="number of frames per sequence (25, 50, 100)", default=100)
    parser.add_argument("--nb_lstm_layers", help="number of deeper lstm layers", default=1)
    parser.add_argument("--use_mask", help="whether to use Mask RCNN mask information", default=1)
    parser.add_argument("--use_bbox", help="whether to use Mask RCNN bbox information", default=1)
    parser.add_argument("--use_prob", help="whether to use Mask RCNN bbox information", default=1)
    parser.add_argument("--use_linear_interpolate", help="whether to linear interpolate loss function", default=1)
    parser.add_argument("--verbose", help="how much information to be printed (smaller=more info)", default=1)
    parser.add_argument("--dropout", help="how much dropout to be applied", default=0.5)
    parser.add_argument("--model", help="model to use (lstm, stack, lstm2, mtl)", default="lstm")
    parser.add_argument("--weights", help="weights to use 0=inception, 1=vgg", default=0)
    parser.add_argument("--store", help="whether to store training profile", default=0)

    args = parser.parse_args()

    SEQ_LEN = int(args.seq_len)
    NB_LSTM_LAYERS = int(args.nb_lstm_layers)
    USE_MASK = bool(args.use_mask)
    USE_BBOX = bool(args.use_bbox)
    USE_PROB = bool(args.use_prob)
    LINEAR_INTERPOLATE = bool(args.use_linear_interpolate)
    VERBOSE = int(args.verbose)
    DROPOUT = float(args.dropout)
    STORE = bool(args.store)
    MODEL_DICT = {
        "lstm": tf_lstm_model.LSTM_VGG16_DEEP,
        "lstm3": tf_lstm_model.LSTM_VGG16_DEEP_USING_MRCNN_INFO,
        "lstm2": tf_lstm_model.LSTM_VGG16_DEEP_USING_MRCNN_INFO2
    }
    MODEL_TO_USE = MODEL_DICT[args.model]
    VGG_OR_INCEPTION = int(args.weights)
    INPUT_SIZE = 4096 * VGG_OR_INCEPTION + 2048 * (1 - VGG_OR_INCEPTION)

    nb_epcs = 100
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    start = time.time()
    tf.reset_default_graph()

    x_train = np.load("loaded_data/lstm/x_train_vgg_features-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    y_train = np.load("loaded_data/lstm/y_train_vgg_features-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    x_test = np.load("loaded_data/lstm/x_test_vgg_features-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    y_test = np.load("loaded_data/lstm/y_test_vgg_features-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    len_train = np.load("loaded_data/lstm/len_train-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    len_test = np.load("loaded_data/lstm/len_test-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))

    mask_train = np.load("loaded_data/lstm/mask_train-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    mask_test = np.load("loaded_data/lstm/mast_test-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    bbox_train = np.load("loaded_data/lstm/bbox_train-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    bbox_test = np.load("loaded_data/lstm/bbox_test-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    prob_train = np.load("loaded_data/lstm/prob_train-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))
    prob_test = np.load("loaded_data/lstm/prob_test-%d-%d.npy" % (SEQ_LEN, VGG_OR_INCEPTION))

    model = MODEL_TO_USE(batch_size=None, seq_len=x_train.shape[1], input_size=INPUT_SIZE,
                         using_bbox=USE_BBOX, using_mask=USE_MASK, using_prob=USE_PROB, dropout=DROPOUT)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("Split into:", x_train.shape, x_test.shape)
    print("Reading took %f seconds" % (time.time() - start))

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for nb_epc in range(nb_epcs):
        start2 = time.time()
        train_batch_gen = util.make_batch(x_train, y_train, len_train, mask_train,
                                          bbox_train, prob_train, batch_size=32)
        val_batch_gen = util.make_batch(x_test, y_test, len_test, mask_test,
                                        bbox_test, prob_test, batch_size=32)
        train_loss = 0
        train_acc = []
        val_loss = 0
        val_acc = []
        while True:
            try:
                x, y, length, m, bb, pr = next(train_batch_gen)
            except StopIteration:
                break
            else:
                _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                        feed_dict={model.input: x, model.label: y,
                                                   model.seq_len: length, model.is_training: True,
                                                   model.mask: m, model.bbox: bb, model.prob_od: pr})
                train_loss += loss
                train_acc.append(acc)

        while True:
            try:
                x, y, length, m, bb, pr = next(val_batch_gen)
            except StopIteration:
                break
            else:
                loss, acc = sess.run([model.loss, model.acc],
                                     feed_dict={model.input: x, model.label: y,
                                                model.seq_len: length, model.is_training: False,
                                                model.mask: m, model.bbox: bb, model.prob_od: pr})
                val_loss += loss
                val_acc.append(acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(np.mean(train_acc))
        val_acc_list.append(np.mean(val_acc))
        if (nb_epc + 1) % VERBOSE == 0 or nb_epc == nb_epcs - 1:
            print("Epoch %d/%d:" % (nb_epc + 1, nb_epcs))
            print("  done in %f seconds" % (time.time() - start2))
            print("  train loss is %f, train acc is %f" % (train_loss, np.mean(train_acc)))
            print("  val loss is %f, val acc is %f" % (val_loss, np.mean(val_acc)))

    if STORE:
        with open("%s-%s.pkl" % ("".join(sys.argv), "train_loss"), 'wb') as f:
            pickle.dump(train_loss_list, f, pickle.HIGHEST_PROTOCOL)
        with open("%s-%s.pkl" % ("".join(sys.argv), "val_loss"), 'wb') as f:
            pickle.dump(val_loss_list, f, pickle.HIGHEST_PROTOCOL)
        with open("%s-%s.pkl" % ("".join(sys.argv), "train_acc"), 'wb') as f:
            pickle.dump(train_acc_list, f, pickle.HIGHEST_PROTOCOL)
        with open("%s-%s.pkl" % ("".join(sys.argv), "val_acc"), 'wb') as f:
            pickle.dump(val_acc_list, f, pickle.HIGHEST_PROTOCOL)

