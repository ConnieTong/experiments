import tensorflow as tf
import numpy as np


class LSTM_VGG16_DEEP:
    def __init__(self, batch_size=None, nb_classes=10, seq_len=50, nb_layers=1, linear_interpolation=False,
                 nb_od_classes=81, dropout=0.5, using_bbox=0, using_mask=0, using_prob=0, input_size=2048):
        state_size = 256

        self.input = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, input_size))
        self.label = tf.placeholder(dtype=tf.int8, shape=(batch_size, nb_classes))
        self.seq_len = tf.placeholder(dtype=tf.int8, shape=(batch_size,))
        self.bbox = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.prob_od = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.is_training = tf.placeholder(dtype=tf.bool)

        with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
            w_fc1 = tf.get_variable(name="w_fc1", shape=(input_size, 512), dtype=tf.float32)
            b_fc1 = tf.get_variable(name="b_fc1", shape=[512], dtype=tf.float32, initializer=tf.zeros_initializer())

        with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
            w_fc2 = tf.get_variable(name="w_fc2", shape=(state_size, nb_classes), dtype=tf.float32)
            b_fc2 = tf.get_variable(name="b_fc2", shape=[nb_classes], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

        self.input2 = tf.reshape(self.input, shape=(-1, input_size))
        self.input2 = tf.matmul(self.input2, w_fc1) + b_fc1
        self.input2 = tf.reshape(self.input2, shape=(-1, seq_len, 512))

        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
                                                             for _ in range(nb_layers)])
            self.output_lstm, _ = tf.nn.dynamic_rnn(self.stacked_lstm, self.input2, dtype=tf.float32,
                                                    sequence_length=self.seq_len)

        self.output_lstm = tf.cond(self.is_training, lambda: tf.nn.dropout(self.output_lstm, dropout),
                                   lambda: self.output_lstm)
        self.output_lstm = tf.reshape(self.output_lstm, shape=(-1, state_size))
        self.output_lstm = tf.matmul(self.output_lstm, w_fc2) + b_fc2
        self.output_lstm = tf.reshape(self.output_lstm, shape=(-1, seq_len, nb_classes))

        seq_mask = tf.sequence_mask(self.seq_len, maxlen=seq_len)
        self.output_lstm = self.output_lstm * tf.cast(tf.reshape(seq_mask, shape=(-1, seq_len, 1)), tf.float32)

        self.loss = []
        linear_interpolation_weights = np.array(range(1, seq_len + 1)) * 2.0 / seq_len / (seq_len + 1)
        for i in range(seq_len):
            loss_at_one_timestep = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_lstm[:, i, :],
                                                                              labels=self.label)
            if linear_interpolation:
                loss_at_one_timestep = loss_at_one_timestep * linear_interpolation_weights[i]
            self.loss.append(loss_at_one_timestep)

        self.loss = tf.stack(self.loss, axis=1) * tf.cast(seq_mask, tf.float32)
        self.loss = tf.reduce_mean(self.loss)
        self.prob = tf.nn.softmax(self.output_lstm, axis=-1)

        if not linear_interpolation:
            self.pred = tf.cast(tf.reduce_sum(tf.argmax(self.output_lstm, 2), 1) / tf.cast(self.seq_len, tf.int64),
                                tf.int64)
        else:
            self.pred = tf.cast(tf.reduce_sum(tf.argmax(self.prob, 2) * linear_interpolation_weights, 1) /
                                tf.cast(self.seq_len, tf.int64), tf.int64)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.label, 1)), tf.float32))
        self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)


class LSTM_VGG16_DEEP_USING_MRCNN_INFO2:
    """
    LSTM2
    """

    def __init__(self, batch_size=None, nb_classes=10, seq_len=50, nb_layers=1, linear_interpolation=False,
                 nb_od_classes=81, using_bbox=True, using_mask=True, using_prob=True, dropout=0.5, input_size=2048):
        state_size = 256
        input_size = 2048
        self.input = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, input_size))
        self.label = tf.placeholder(dtype=tf.int8, shape=(batch_size, nb_classes))
        self.seq_len = tf.placeholder(dtype=tf.int8, shape=(batch_size,))
        self.bbox = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.prob_od = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.is_training = tf.placeholder(dtype=tf.bool)

        with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
            w_fc1 = tf.get_variable(name="w_fc1", shape=(input_size, 512), dtype=tf.float32)
            b_fc1 = tf.get_variable(name="b_fc1", shape=[512], dtype=tf.float32, initializer=tf.zeros_initializer())

        with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
            w_fc2 = tf.get_variable(name="w_fc2", shape=(state_size + 8, nb_classes),
                                    dtype=tf.float32)
            b_fc2 = tf.get_variable(name="b_fc2", shape=[nb_classes], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

        with tf.variable_scope("fc3", reuse=tf.AUTO_REUSE):
            w_fc3 = tf.get_variable(name="w_fc3", shape=(nb_od_classes, 16), dtype=tf.float32)
            b_fc3 = tf.get_variable(name="b_fc3", shape=[16], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

        self.input2 = tf.reshape(self.input, shape=(-1, input_size))
        self.input2 = tf.matmul(self.input2, w_fc1) + b_fc1
        self.input2 = tf.reshape(self.input2, shape=(-1, seq_len, 512))

        self.mrcnn_info = []
        if using_bbox:
            self.bbox2 = tf.reshape(self.bbox, shape=(-1, nb_od_classes))
            self.bbox2 = tf.matmul(self.bbox2, w_fc3) + b_fc3
            self.bbox2 = tf.reshape(self.bbox2, shape=(-1, seq_len, 16))
            self.mrcnn_info.append(self.bbox2)
        if using_mask:
            self.mask2 = tf.reshape(self.mask, shape=(-1, nb_od_classes))
            self.mask2 = tf.matmul(self.mask2, w_fc3) + b_fc3
            self.mask2 = tf.reshape(self.mask2, shape=(-1, seq_len, 16))
            self.mrcnn_info.append(self.mask2)
        if using_prob:
            self.prob_od2 = tf.reshape(self.prob_od, shape=(-1, nb_od_classes))
            self.prob_od2 = tf.matmul(self.prob_od2, w_fc3) + b_fc3
            self.prob_od2 = tf.reshape(self.prob_od2, shape=(-1, seq_len, 16))
            self.mrcnn_info.append(self.prob_od2)
        if len(self.mrcnn_info) > 0:
            self.mrcnn_info = tf.concat(self.mrcnn_info, axis=-1)

        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
                                                             for _ in range(nb_layers)])
            self.output_lstm, _ = tf.nn.dynamic_rnn(self.stacked_lstm, self.input2, dtype=tf.float32,
                                                    sequence_length=self.seq_len)

        with tf.variable_scope("lstm2", reuse=tf.AUTO_REUSE):
            self.stacked_lstm2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(8, state_is_tuple=True)
                                                              for _ in range(1)])
            self.output_lstm2, _ = tf.nn.dynamic_rnn(self.stacked_lstm2, self.mrcnn_info, dtype=tf.float32,
                                                     sequence_length=self.seq_len)
        self.output_lstm = tf.cond(self.is_training, lambda: tf.nn.dropout(self.output_lstm, dropout),
                                   lambda: self.output_lstm)
        self.output_lstm = tf.concat([self.output_lstm, self.output_lstm2], axis=-1)
        self.output_lstm = tf.reshape(self.output_lstm, shape=(-1, state_size + 8))
        self.output_lstm = tf.matmul(self.output_lstm, w_fc2) + b_fc2
        self.output_lstm = tf.cond(self.is_training, lambda: tf.nn.dropout(self.output_lstm, dropout),
                                   lambda: self.output_lstm)
        self.output_lstm = tf.reshape(self.output_lstm, shape=(-1, seq_len, nb_classes))

        seq_mask = tf.sequence_mask(self.seq_len, maxlen=seq_len)
        self.output_lstm = self.output_lstm * tf.cast(tf.reshape(seq_mask, shape=(-1, seq_len, 1)), tf.float32)

        self.loss = []
        linear_interpolation_weights = np.array(range(1, seq_len + 1)) * 2.0 / seq_len / (seq_len + 1)
        for i in range(seq_len):
            loss_at_one_timestep = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_lstm[:, i, :],
                                                                              labels=self.label)
            if linear_interpolation:
                loss_at_one_timestep = loss_at_one_timestep * linear_interpolation_weights[i]
            self.loss.append(loss_at_one_timestep)

        self.loss = tf.stack(self.loss, axis=1) * tf.cast(seq_mask, tf.float32)
        self.loss = tf.reduce_mean(self.loss)

        self.prob = tf.nn.softmax(self.output_lstm, axis=-1)

        if not linear_interpolation:
            self.pred = tf.cast(tf.reduce_sum(tf.argmax(self.output_lstm, 2), 1) / tf.cast(self.seq_len, tf.int64),
                                tf.int64)
        else:
            self.pred = tf.cast(tf.reduce_sum(tf.argmax(self.prob, 2) * linear_interpolation_weights, 1) /
                                tf.cast(self.seq_len, tf.int64), tf.int64)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.label, 1)), tf.float32))
        self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)


class LSTM_VGG16_DEEP_USING_MRCNN_INFO:
    """
    LSTM
    """

    def __init__(self, batch_size=None, nb_classes=10, seq_len=50, nb_layers=1, linear_interpolation=False,
                 nb_od_classes=81, using_bbox=True, using_mask=True, using_prob=True, dropout=0.5, input_size=2048):
        state_size = 256
        input_size = 2048
        self.input = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, input_size))
        self.label = tf.placeholder(dtype=tf.int8, shape=(batch_size, nb_classes))
        self.seq_len = tf.placeholder(dtype=tf.int8, shape=(batch_size,))
        self.bbox = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.prob_od = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, nb_od_classes))
        self.is_training = tf.placeholder(dtype=tf.bool)

        with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
            w_fc1 = tf.get_variable(name="w_fc1", shape=(input_size, 512), dtype=tf.float32)
            b_fc1 = tf.get_variable(name="b_fc1", shape=[512], dtype=tf.float32, initializer=tf.zeros_initializer())

        with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
            w_fc2 = tf.get_variable(name="w_fc2", shape=(state_size, nb_classes),
                                    dtype=tf.float32)
            b_fc2 = tf.get_variable(name="b_fc2", shape=[nb_classes], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

        with tf.variable_scope("fc3", reuse=tf.AUTO_REUSE):
            w_fc3 = tf.get_variable(name="w_fc3", shape=(nb_od_classes, 16), dtype=tf.float32)
            b_fc3 = tf.get_variable(name="b_fc3", shape=[16], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

        self.input2 = tf.reshape(self.input, shape=(-1, input_size))
        self.input2 = tf.matmul(self.input2, w_fc1) + b_fc1
        self.input2 = tf.reshape(self.input2, shape=(-1, seq_len, 512))

        self.mrcnn_info = []
        if using_bbox:
            self.bbox2 = tf.reshape(self.bbox, shape=(-1, nb_od_classes))
            self.bbox2 = tf.matmul(self.bbox2, w_fc3) + b_fc3
            self.bbox2 = tf.reshape(self.bbox2, shape=(-1, seq_len, 16))
            self.mrcnn_info.append(self.bbox2)
        if using_mask:
            self.mask2 = tf.reshape(self.mask, shape=(-1, nb_od_classes))
            self.mask2 = tf.matmul(self.mask2, w_fc3) + b_fc3
            self.mask2 = tf.reshape(self.mask2, shape=(-1, seq_len, 16))
            self.mrcnn_info.append(self.mask2)
        if using_prob:
            self.prob_od2 = tf.reshape(self.prob_od, shape=(-1, nb_od_classes))
            self.prob_od2 = tf.matmul(self.prob_od2, w_fc3) + b_fc3
            self.prob_od2 = tf.reshape(self.prob_od2, shape=(-1, seq_len, 16))
            self.mrcnn_info.append(self.prob_od2)
        if len(self.mrcnn_info) > 0:
            self.mrcnn_info.insert(0, self.input2)
            self.mrcnn_info = tf.concat(self.mrcnn_info, axis=-1)
        else:
            self.mrcnn_info = self.input2

        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
                                                             for _ in range(nb_layers)])
            self.output_lstm, _ = tf.nn.dynamic_rnn(self.stacked_lstm, self.mrcnn_info, dtype=tf.float32,
                                                    sequence_length=self.seq_len)

        self.output_lstm = tf.cond(self.is_training, lambda: tf.nn.dropout(self.output_lstm, dropout),
                                   lambda: self.output_lstm)
        self.output_lstm = tf.reshape(self.output_lstm, shape=(-1, state_size))
        self.output_lstm = tf.matmul(self.output_lstm, w_fc2) + b_fc2
        self.output_lstm = tf.cond(self.is_training, lambda: tf.nn.dropout(self.output_lstm, dropout),
                                   lambda: self.output_lstm)
        self.output_lstm = tf.reshape(self.output_lstm, shape=(-1, seq_len, nb_classes))

        seq_mask = tf.sequence_mask(self.seq_len, maxlen=seq_len)
        self.output_lstm = self.output_lstm * tf.cast(tf.reshape(seq_mask, shape=(-1, seq_len, 1)), tf.float32)

        self.loss = []
        linear_interpolation_weights = np.array(range(1, seq_len + 1)) * 2.0 / seq_len / (seq_len + 1)
        for i in range(seq_len):
            loss_at_one_timestep = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_lstm[:, i, :],
                                                                              labels=self.label)
            if linear_interpolation:
                loss_at_one_timestep = loss_at_one_timestep * linear_interpolation_weights[i]
            self.loss.append(loss_at_one_timestep)

        self.loss = tf.stack(self.loss, axis=1) * tf.cast(seq_mask, tf.float32)
        self.loss = tf.reduce_mean(self.loss)
        self.prob = tf.nn.softmax(self.output_lstm, axis=-1)

        if not linear_interpolation:
            self.pred = tf.cast(tf.reduce_sum(tf.argmax(self.prob, 2), 1) / tf.cast(self.seq_len, tf.int64),
                                tf.int64)
        else:
            self.pred = tf.cast(tf.reduce_sum(tf.argmax(self.prob, 2) * linear_interpolation_weights, 1) /
                                tf.cast(self.seq_len, tf.int64), tf.int64)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.label, 1)), tf.float32))
        self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)


if __name__ == '__main__':
    model = LSTM_VGG16_DEEP_USING_MRCNN_INFO(batch_size=32, linear_interpolation=True)
