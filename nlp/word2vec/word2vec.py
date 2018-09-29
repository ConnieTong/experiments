import os
import tensorflow as tf
import numpy as np
import tqdm
import data_preprocessing as dp
import math
import evaluation
from tensorboard.plugins import projector


"""
CPU job
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
Parameters
"""
BATCH_SIZE = 32  # number of samples per batch
EMBEDDING_SIZE = 128  # dimension of the embedding vector
WINDOW_SIZE = 1  # how many words to consider left and right
NEG_SAMPLES = 64  # number of negative examples to sample
DOMAIN_LIMIT = 5000  # limit per domain to read
NB_EPOCHS = 10  # number of epochs to train
LEARNING_RATE = 0.1  # learning rate to optimize

"""
Directories
"""
TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
TMP_DIR = "/tmp/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"


"""
Read data and write them to disk, only run for a first time. Comment the line below if you want to load them instead.
"""
dp.write_dataset(domain_limit=DOMAIN_LIMIT, skip_window=WINDOW_SIZE)


"""
Load data from disk.
"""
data, counter, W2I, I2W, unigram_table = dp.load_dataset()
VOCABULARY_SIZE = len(W2I)  # The most N word to consider in the dictionary
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
questions = dp.read_analogies(ANALOGIES_FILE, W2I)
print("Load data done")
print("Data size: %d samples with %d unique words" % (data.shape[0], len(W2I)))


"""
Skip gram model
Below is a long sequence of graph nodes for the skip gram model. The flow is basically from the (inputs, labels)
to the positive entropy loss (posEnt) and negative entropy loss (negEnt). To calculate posEnt, one just needs to
calculate posLog = inputs*pos_outputs and then forward it to sigmoid cross entropy with positive labels. 
Similarly, to negEnt, one first calculates negLog = inputs*neg_outputs ==> sigmoid cross entropy with 
negative labels. While pos_outputs are lookups of vector 'labels' on embedding tables, neg_outputs are lookups of
vector 'sampled_indices' which have to be determined by a candidate sampling function.
"""
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int64, shape=[BATCH_SIZE])  # input words
    train_labels = tf.placeholder(tf.int64, shape=[BATCH_SIZE])  # target words
    sampled_indices = tf.placeholder(tf.int64, shape=[BATCH_SIZE, NEG_SAMPLES])  # negative words
    labels_matrix = tf.reshape(tf.cast(train_labels, dtype=tf.int64), [BATCH_SIZE, 1])  # target words (different shape)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  # validating input words

    # embedding table
    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE]))

    # calculate entropy loss for positive samples
    input_emb = tf.nn.embedding_lookup(embeddings, train_inputs)  # inp [batch size, embedding size]
    output_emb = tf.nn.embedding_lookup(embeddings, train_labels)  # positive outputs [batch size, embedding size]
    pos_log = tf.reduce_sum(tf.multiply(input_emb, output_emb), 1)  # inputs * positive outputs
    pos_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_log), logits=pos_log)
    pos_ent = tf.reduce_sum(pos_ent)  # sigmoid cross entropy with positive labels

    # calculate entropy loss for negative samples
    input_emb2 = tf.reshape(input_emb, [-1, 1, EMBEDDING_SIZE])  # inp2, same content as inp but different shape
    output_emb2 = tf.nn.embedding_lookup(embeddings, sampled_indices)  # negative outputs
    neg_log = tf.matmul(output_emb2, input_emb2, transpose_b=True)  # negWeights * negative outputs
    neg_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_log), logits=neg_log)
    neg_ent = tf.reduce_sum(neg_ent)  # sigmoid cross entropy with negative labels

    # total loss
    loss = (pos_ent + neg_ent) / BATCH_SIZE
    tf.summary.scalar('loss', loss)

    # optimizing op
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    # normalize
    normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)

    # valid information
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # merge all summaries
    merged = tf.summary.merge_all()

    # create a saver
    saver = tf.train.Saver()

    # analogy evaluator
    evaluator = evaluation.Evaluation(normalized_embeddings, W2I, questions)

    # global init
    init = tf.global_variables_initializer()


"""
Training procedure
"""
num_steps = int(math.ceil(data.shape[0] / BATCH_SIZE)) * NB_EPOCHS  # num steps to train
data_index = 0  # determine which batch to extract
average_loss = 0  # overall loss value
nth_epoch = 1
with tf.Session(graph=graph) as session:
    # open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)

    # we must initialize all variables before we use them.
    init.run()
    print('Initialized')

    bar = tqdm.tqdm(range(1, num_steps+1))
    for step in bar:
        batch_inputs, batch_labels, exhausted = dp.generate_batch(BATCH_SIZE, data_index, data)

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        if exhausted:
            data_index = 0
            print("Epoch %d done" % nth_epoch)
            nth_epoch += 1
            continue
        else:
            data_index += BATCH_SIZE

        # select negative samples
        all_sampled = []
        for gh in range(BATCH_SIZE):
            neg_sample = dp.select_negative_samples(unigram_table, NEG_SAMPLES, batch_labels[gh])
            all_sampled.append(neg_sample)
        neg_samples = np.stack(all_sampled)

        # optimize
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict={train_inputs: batch_inputs, train_labels: batch_labels, sampled_indices: neg_samples},
            run_metadata=run_metadata)
        average_loss += loss_val

        # add returned summaries to writer in each step.
        writer.add_summary(summary, step)

        if step % 50000 == 0:
            # some nearest words
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = I2W[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = I2W[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

        if step % 5000:
            # analogy accuracy
            evaluator.eval(session)
            print("avg loss: " + str(average_loss / step))

        # add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step %d' % step)

    # final analogy accuracy
    evaluator.eval(session)

    # normalize embeddings
    final_embeddings = normalized_embeddings.eval()

    # save vectors
    dp.save_vectors(final_embeddings)

    # Write corresponding labels for the embeddings.
    with open(TMP_DIR + 'metadata.tsv', 'w') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(I2W[i] + '\n')

    # save the model for checkpoints
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

    # create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(TMP_DIR, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
