import tensorflow as tf


class Model:
    def __init__(self, nb_senses_max, batch_size, nb_target_words, is_training=True):
        """
        Bidirectional LSTM with hidden layer
        :param nb_senses_max: max number of possible senses for a target word
        :param batch_size:
        :param nb_target_words: max number of target words in training data
        :param is_training: whether to apply dropout (set to False when test)
        """
        state_size = 400  # lstm state size
        hidden_size = 200  # hidden layer size

        self.inputs_fw = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, 300))  # forward pass of sentence
        self.inputs_bw = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, 300))  # backward pass of sentence

        # actual length of sentence (forward and backward) to ignore the padding
        self.length_fw = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
        self.length_bw = tf.placeholder(dtype=tf.int32, shape=(batch_size,))

        # indices of the last words in forward and backward pass (this is for extracting the right lstm's h outputs
        # exactly at the end of each pass). Basically, these args are same as length_fw and length_bw (in terms of
        # meaning) but are modified to fit in TF.
        self.output_fw_index = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.output_bw_index = tf.placeholder(dtype=tf.int32, shape=(None, None))

        # sense targets
        self.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, None))

        # number of possible senses for each target word (this is for ignoring the padding in softmax params)
        self.nb_senses = tf.placeholder(dtype=tf.int32, shape=(batch_size,))

        # IDs of target words in the vocab (this is for looking up the right softmax params because each target word
        # has its own softmax)
        self.target_word_id = tf.placeholder(dtype=tf.int32, shape=(batch_size,))

        # softmax params (a big matrix containing all the softmax for all the target words is initialized, then based
        # on the ID, each target word then gets its own softmax, then based on the number of possible senses for that
        # word, padding will be stripped)
        with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
            # Big matrices
            self.w_target_mat = tf.get_variable("w_target", shape=(nb_target_words, hidden_size, nb_senses_max),
                                                initializer=tf.random_uniform_initializer(0, 1))
            self.b_target_mat = tf.get_variable("b_target", shape=(nb_target_words, 1, nb_senses_max),
                                                initializer=tf.constant_initializer(0.0))

            # Look up based on target words' ID
            w_target = tf.nn.embedding_lookup(self.w_target_mat, self.target_word_id)
            b_target = tf.nn.embedding_lookup(self.b_target_mat, self.target_word_id)

        # Hidden layers to shrink down lstm's h outputs before going to softmax layer
        with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
            self.w_hidden = tf.get_variable("w_hidden", shape=(state_size, hidden_size),
                                            initializer=tf.random_uniform_initializer(0, 1))
            self.b_hidden = tf.get_variable("b_hidden", shape=(1, hidden_size),
                                            initializer=tf.constant_initializer(0.0))

        # lstm's forward pass
        with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
            self.cell_fw = tf.contrib.rnn.LSTMCell(200, state_is_tuple=True)
            state_fw = self.cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
            self.output_fw, state_fw = tf.nn.dynamic_rnn(self.cell_fw, self.inputs_fw, dtype=tf.float32,
                                                         sequence_length=self.length_fw, initial_state=state_fw)

        # lstm's backward pass
        with tf.variable_scope("backward", reuse=tf.AUTO_REUSE):
            self.cell_bw = tf.contrib.rnn.LSTMCell(200, state_is_tuple=True)
            state_bw = self.cell_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
            self.output_bw, state_bw = tf.nn.dynamic_rnn(self.cell_bw, self.inputs_bw, dtype=tf.float32,
                                                         sequence_length=self.length_bw, initial_state=state_bw)

        # select the last h outputs in the sequence and concat them
        self.out1 = tf.gather_nd(self.output_fw, self.output_fw_index)
        self.out2 = tf.gather_nd(self.output_bw, self.output_bw_index)
        self.lstm_out = tf.concat([self.out1, self.out2], axis=-1)

        # Dropout
        if is_training:
            self.lstm_out = tf.nn.dropout(self.lstm_out, keep_prob=0.5)
            self.w_hidden = tf.nn.dropout(self.w_hidden, keep_prob=0.5)

        # LSTM => Logits => Loss + Prob
        mask = tf.sequence_mask(self.nb_senses, maxlen=nb_senses_max)
        self.lstm_out = tf.matmul(self.lstm_out, self.w_hidden) + self.b_hidden
        self.lstm_out = tf.reshape(self.lstm_out, (-1, 1, 200))
        self.logits = tf.matmul(self.lstm_out, w_target) + b_target
        self.logits = tf.reshape(self.logits, (batch_size, -1)) * tf.cast(mask, tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

        # Train
        with tf.variable_scope("train_op", reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)

        # Prediction and accuracy (mostly for testing)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.prob, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.targets, 1)), tf.float32))


class Model2_hp:
    def __init__(self, nb_senses_max, batch_size, nb_target_words, is_training=True):
        """
        Bidirectional LSTM with hidden layer
        :param nb_senses_max: max number of possible senses for a target word
        :param batch_size:
        :param nb_target_words: max number of target words in training data
        :param is_training: whether to apply dropout (set to False when test)
        """
        state_size = 400  # lstm state size
        hidden_size = 200  # hidden layer size

        self.inputs_fw = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, 300))  # forward pass of sentence
        self.inputs_bw = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, 300))  # backward pass of sentence

        # actual length of sentence (forward and backward) to ignore the padding
        self.length_fw = tf.placeholder(dtype=tf.int16, shape=(batch_size,))
        self.length_bw = tf.placeholder(dtype=tf.int16, shape=(batch_size,))

        # indices of the last words in forward and backward pass (this is for extracting the right lstm's h outputs
        # exactly at the end of each pass). Basically, these args are same as length_fw and length_bw (in terms of
        # meaning) but are modified to fit in TF.
        self.output_fw_index = tf.placeholder(dtype=tf.int16, shape=(None, None))
        self.output_bw_index = tf.placeholder(dtype=tf.int16, shape=(None, None))

        # sense targets
        self.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, None))

        # number of possible senses for each target word (this is for ignoring the padding in softmax params)
        self.nb_senses = tf.placeholder(dtype=tf.int16, shape=(batch_size,))

        # IDs of target words in the vocab (this is for looking up the right softmax params because each target word
        # has its own softmax)
        self.target_word_id = tf.placeholder(dtype=tf.int16, shape=(batch_size,))

        # softmax params (a big matrix containing all the softmax for all the target words is initialized, then based
        # on the ID, each target word then gets its own softmax, then based on the number of possible senses for that
        # word, padding will be stripped)
        with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
            # Big matrices
            self.w_target_mat = tf.get_variable("w_target", shape=(nb_target_words, hidden_size, nb_senses_max),
                                                initializer=tf.random_uniform_initializer(0, 1))
            self.b_target_mat = tf.get_variable("b_target", shape=(nb_target_words, 1, nb_senses_max),
                                                initializer=tf.constant_initializer(0.0))

            # Look up based on target words' ID
            w_target = tf.nn.embedding_lookup(self.w_target_mat, self.target_word_id)
            b_target = tf.nn.embedding_lookup(self.b_target_mat, self.target_word_id)

        # Hidden layers to shrink down lstm's h outputs before going to softmax layer
        with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
            self.w_hidden = tf.get_variable("w_hidden", shape=(state_size, hidden_size),
                                            initializer=tf.random_uniform_initializer(0, 1))
            self.b_hidden = tf.get_variable("b_hidden", shape=(1, hidden_size),
                                            initializer=tf.constant_initializer(0.0))

        # lstm's forward pass
        with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
            self.cell_fw = tf.contrib.rnn.LSTMCell(200, state_is_tuple=True)
            state_fw = self.cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
            self.output_fw, state_fw = tf.nn.dynamic_rnn(self.cell_fw, self.inputs_fw, dtype=tf.float32,
                                                         sequence_length=self.length_fw, initial_state=state_fw)

        # lstm's backward pass
        with tf.variable_scope("backward", reuse=tf.AUTO_REUSE):
            self.cell_bw = tf.contrib.rnn.LSTMCell(200, state_is_tuple=True)
            state_bw = self.cell_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
            self.output_bw, state_bw = tf.nn.dynamic_rnn(self.cell_bw, self.inputs_bw, dtype=tf.float32,
                                                         sequence_length=self.length_bw, initial_state=state_bw)

        # select the last h outputs in the sequence and concat them
        self.out1 = tf.gather_nd(self.output_fw, self.output_fw_index)
        self.out2 = tf.gather_nd(self.output_bw, self.output_bw_index)
        self.lstm_out = tf.concat([self.out1, self.out2], axis=-1)

        # Dropout
        if is_training:
            self.lstm_out = tf.nn.dropout(self.lstm_out, keep_prob=0.5)
            self.w_hidden = tf.nn.dropout(self.w_hidden, keep_prob=0.5)

        # LSTM => Logits => Loss + Prob
        mask = tf.sequence_mask(self.nb_senses, maxlen=nb_senses_max)
        self.lstm_out = tf.matmul(self.lstm_out, self.w_hidden) + self.b_hidden
        self.lstm_out = tf.reshape(self.lstm_out, (-1, 1, 200))
        self.logits = tf.matmul(self.lstm_out, w_target) + b_target
        self.logits = tf.reshape(self.logits, (batch_size, -1)) * tf.cast(mask, tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

        # Train
        with tf.variable_scope("train_op", reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdagradOptimizer(0.1).minimize(self.loss)

        # Prediction and accuracy (mostly for testing)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.prob, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.targets, 1)), tf.float32))



if __name__ == "__main__":
    tf.reset_default_graph()
    m = Model2(10, 32, 10)