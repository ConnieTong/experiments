import tensorflow as tf


class Model_span_based:
    """
    LSTM cell model (only 1 layer is possible)
    """
    def __init__(self, batch_size=None, feature_len=251, nb_classes=54, nb_layers=0):
        state_size = 64

        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, None, feature_len))  # batch size x seq len x feature len
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=(batch_size, None))  # batch size x seq len
        self.actual_lengths = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
        self.max_length_in_batch = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
            self.cell_fw = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
            self.output_fw, state_fw = tf.nn.dynamic_rnn(self.cell_fw, self.input, dtype=tf.float32,
                                                         sequence_length=self.actual_lengths)

        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            w_target = tf.get_variable(name="w_target", shape=[state_size, nb_classes], dtype=tf.float32)

            b_target = tf.get_variable(name="b_target", shape=[nb_classes], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

        self.output_fw = tf.reshape(self.output_fw, [-1, state_size])
        self.logits = tf.matmul(self.output_fw, w_target) + b_target
        self.logits = tf.reshape(self.logits, [-1, self.max_length_in_batch, nb_classes])

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.actual_lengths)

        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params,
                                                                              self.actual_lengths)

        # Training ops.
        self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)


class Model_span_based_deeper_lstm:
    """
    MultiRNN cell model (same as LSTM cell model but more layers are possible)
    """
    def __init__(self, batch_size=None, feature_len=251, nb_classes=54, nb_layers=2):
        state_size = 64

        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, None, feature_len))  # batch size x seq len x feature len
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=(batch_size, None))  # batch size x seq len
        self.actual_lengths = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
        self.max_length_in_batch = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
                                                             for _ in range(nb_layers)])
            self.output_fw, _ = tf.nn.dynamic_rnn(self.stacked_lstm, self.input, dtype=tf.float32,
                                                  sequence_length=self.actual_lengths)

        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            w_target = tf.get_variable(name="w_target", shape=[state_size, nb_classes], dtype=tf.float32)

            b_target = tf.get_variable(name="b_target", shape=[nb_classes], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

        self.output_fw = tf.reshape(self.output_fw, [-1, state_size])
        self.logits = tf.matmul(self.output_fw, w_target) + b_target
        self.logits = tf.reshape(self.logits, [-1, self.max_length_in_batch, nb_classes])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.actual_lengths)

        self.loss = tf.reduce_mean(-self.log_likelihood)

        # Compute the viterbi sequence and score (used for prediction and test time).
        self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params,
                                                                              self.actual_lengths)

        # Training ops.
        self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)


class Model_span_based_bi_lstm:
    """
    Bi LSTM model
    """
    def __init__(self, batch_size=None, feature_len=251, nb_classes=54, nb_layers=0):
        state_size = 100
        hidden1_size = 200
        hidden2_size = 100

        self.input_fw = tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, None, feature_len))  # batch size x seq len x feature len
        self.input_bw = tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, None, feature_len))  # batch size x seq len x feature len
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=(batch_size, None))  # batch size x seq len
        self.actual_lengths = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
        self.max_length_in_batch = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope("hidden1", reuse=tf.AUTO_REUSE):
            w_hidden1 = tf.get_variable(name="w_hidden1", shape=[feature_len, hidden1_size], dtype=tf.float32)
            b_hidden1 = tf.get_variable(name="b_hidden1", shape=[hidden1_size], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

        with tf.variable_scope("hidden2", reuse=tf.AUTO_REUSE):
            w_hidden2 = tf.get_variable(name="w_hidden2", shape=[state_size*2, hidden2_size], dtype=tf.float32)
            b_hidden2 = tf.get_variable(name="b_hidden2", shape=[hidden2_size], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

        with tf.variable_scope("target", reuse=tf.AUTO_REUSE):
            w_target = tf.get_variable(name="w_target", shape=[hidden2_size, nb_classes], dtype=tf.float32)
            b_target = tf.get_variable(name="b_target", shape=[nb_classes], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

        self.input_fw2 = tf.tanh(
            tf.reshape(tf.matmul(tf.reshape(self.input_fw, [-1, feature_len]), w_hidden1) + b_hidden1,
                       [-1, self.max_length_in_batch, hidden1_size]))
        self.input_bw2 = tf.tanh(
            tf.reshape(tf.matmul(tf.reshape(self.input_bw, [-1, feature_len]), w_hidden1) + b_hidden1,
                       [-1, self.max_length_in_batch, hidden1_size]))

        with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
            self.cell_fw = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
            self.output_fw, _ = tf.nn.dynamic_rnn(self.cell_fw, self.input_fw2, dtype=tf.float32,
                                                  sequence_length=self.actual_lengths)

        with tf.variable_scope("backward", reuse=tf.AUTO_REUSE):
            self.cell_bw = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
            self.output_bw, _ = tf.nn.dynamic_rnn(self.cell_bw, self.input_bw2, dtype=tf.float32,
                                                  sequence_length=self.actual_lengths)

        self.output = tf.concat([self.output_fw, self.output_bw], axis=-1)
        self.output = tf.tanh(tf.reshape(tf.matmul(tf.reshape(self.output, [-1, state_size*2]), w_hidden2) + b_hidden2,
                                         shape=[-1, self.max_length_in_batch, hidden2_size]))
        self.logits = tf.reshape(tf.matmul(tf.reshape(self.output, [-1, hidden2_size]), w_target) + b_target,
                                 shape=[-1, self.max_length_in_batch, nb_classes])

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.actual_lengths)

        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params,
                                                                              self.actual_lengths)

        # Training ops.
        self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)


if __name__ == '__main__':
    tf.reset_default_graph()
    m = Model_span_based_bi_lstm(batch_size=32)
    merged = tf.summary.merge_all()
    sess = tf.Session()
    tf.summary.FileWriter("tf_logs", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
