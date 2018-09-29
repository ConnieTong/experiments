import math
import os
import random
import numpy as np
import pickle
import string
import re
import collections


PUNCTUATION = string.punctuation  # punctuation to be removed


def generate_batch(batch_size, first_index, data):
    """
    Generate a batch from a numpy array
    :param batch_size: size of the batch
    :param first_index: the current index of the batch
    :param data: the numpy array
    :return: the batch and a bool flag indicating if all batches have been extracted
    """
    train_data = data[first_index: first_index+batch_size, 0]
    labels = data[first_index: first_index+batch_size, 1]
    exhausted = False
    if first_index + batch_size > data.shape[0]:
        exhausted = True
    return train_data, labels, exhausted


def preprocess_word(a_word):
    """
    Preprocess a word
    :param a_word: input word
    :return: a preprocessed word or empty string if the input word is non ascii
    """
    # remove all the punctuations from the input word
    a_word = a_word.translate(None, PUNCTUATION).strip().lower()

    # check if input word is ascii
    try:
        a_word.decode('ascii')
    except UnicodeDecodeError:
        return ""

    # check if input contains numbers
    for c in a_word:
        if c in "0123456789":
            return "NUM"
    return a_word


def build_dictionary(a_counter, min_count=5):
    """
    Build a dictionary from a list of words
    :param a_counter: frequency map for words
    :param min_count: minimum frequency of a word to be appeared in the dictionary
    :return: word2int, int2word
    """
    word2int = {}
    sorted_keys = sorted(a_counter.keys(), key=lambda x: a_counter[x], reverse=True)
    for k in sorted_keys:
        if k not in word2int and a_counter[k] >= min_count and k != "UNK":
            word2int[k] = len(word2int)
        else:
            a_counter["UNK"] += 1
    word2int["UNK"] = len(word2int)
    print("Dataset has %d unique words" % len(word2int), max(word2int.values()))
    for k in range(len(word2int)):
        assert k in word2int.values()
    int2word = dict(zip(word2int.values(), word2int.keys()))
    return word2int, int2word


def read_data(directory="dataset/DATA/TRAIN", domain_limit=-1):
    """
    Read the dataset into a list of preprocessed words
    :param directory: dir of training data
    :param domain_limit: limitation per domain
    :return: list of words
    """
    folders = [x[0] for x in os.walk(directory) if len(x[0].split("/")) == 4]
    for domain in folders:
        limit = domain_limit
        all_files = [os.path.join(domain, f) for f in os.listdir(domain) if os.path.isfile(os.path.join(domain, f))]
        for i in range(len(all_files)):
            with open(all_files[i], 'r') as content_file:
                content = content_file.read()

            # there is domain limit and limit is not reached
            if limit > 0 and domain_limit != -1:
                new_vocab = [token for token in map(preprocess_word, re.split("[ \\n\\t]", content))
                             if token != "" and token != " "][:limit]
                limit -= len(new_vocab)
                yield new_vocab

            # there is no domain limit
            elif domain_limit == -1:
                new_vocab = [token for token in map(preprocess_word, re.split("[ \\n\\t]", content))
                             if token != "" and token != " "]
                yield new_vocab

            # there is domain limit and limit is reached
            elif domain_limit != -1 and limit <= 0:
                break


def write_dataset(domain_limit=-1, skip_window=1, sub="prob", min_count=5):
    """
    Read the dataset for word2vec task and save necessary variables to disk
    :param domain_limit: limitation per domain
    :param skip_window: window size
    :param sub: "prob" or None to do subsample to not
    :param min_count: minimum frequency to appear in vocab
    :return:
    """
    counter = {"UNK": 0}
    data_reader = read_data(domain_limit=domain_limit)

    # count freq
    while True:
        try:
            batch = data_reader.next()
            for w in batch:
                if w not in counter:
                    counter[w] = 0
                else:
                    counter[w] += 1
        except StopIteration:
            break

    # make dict
    word2int, int2word = build_dictionary(counter, min_count=min_count)

    with open('data/counter' + '.pkl', 'wb') as f:
        pickle.dump(counter, f, pickle.HIGHEST_PROTOCOL)
    with open('data/w2i' + '.pkl', 'wb') as f:
        pickle.dump(word2int, f, pickle.HIGHEST_PROTOCOL)
    with open('data/i2w' + '.pkl', 'wb') as f:
        pickle.dump(int2word, f, pickle.HIGHEST_PROTOCOL)

    # write dataset
    data_reader = read_data(domain_limit=domain_limit)
    first_time = {1: "w", 0: "a"}
    first_time_track = 1
    while True:
        try:
            batch = data_reader.next()
            data_processor = build_dataset(batch, word2int, int2word, counter,
                                           skip_window=skip_window, subsample_type=sub)
            while True:
                try:
                    sample = data_processor.next()
                    assert word2int[int2word[sample[0]]] == sample[0]
                    assert word2int[int2word[sample[1]]] == sample[1]
                    with open("data/DATA_words.txt", first_time[first_time_track]) as myfile:
                        myfile.write("%s %s %d %d\n" % (int2word[sample[0]], int2word[sample[1]],
                                                        counter[int2word[sample[0]]], counter[int2word[sample[1]]]))
                    with open("data/DATA.txt", first_time[first_time_track]) as myfile:
                        myfile.write("%d %d\n" % (sample[0], sample[1]))
                    first_time_track = 0
                except StopIteration:
                    break
        except StopIteration:
            break


def build_dataset(vocab, w2i, i2w, a_counter, skip_window=1, subsample_type="prob", subsample_const=1e-3):
    """
    Build a data generator that returns a pair of words each time with no restriction on word context (a word can
    pair with any word within the window)
    :param vocab: list of words
    :param w2i: word2int
    :param i2w: int2word
    :param a_counter: counter word frequency
    :param skip_window: window size
    :param subsample_type: None or prob to subsample frequent words
    :param subsample_const: const for subsampling
    :return: a generator
    """
    for word_index, input_word in enumerate(vocab):
        rand_skip_window = random.choice(range(1, skip_window+1))
        for target_word in vocab[max(word_index - rand_skip_window, 0): min(word_index + rand_skip_window,
                                                                            len(vocab)) + 1]:
            if target_word != input_word:
                inp = w2i.get(input_word, w2i["UNK"])
                target = w2i.get(target_word, w2i["UNK"])
                if subsample_type == "prob":
                    freq = a_counter[i2w[inp]]
                    keep_prob = math.sqrt(freq/(subsample_const*len(vocab))+1)*(subsample_const*len(vocab)/freq)
                    if random.random() > keep_prob:
                        continue
                yield [inp, target]


def load_dataset():
    """
    Load the files saved by write_dataset
    :return: dataset (word pairs), a counter, word2int, int2word, unigram table
    """
    # load counter and word2int, int2word dictionaries from disk
    with open('data/counter' + '.pkl', 'rb') as f:
        a_counter = pickle.load(f)
    with open('data/w2i' + '.pkl', 'rb') as f:
        w2i = pickle.load(f)
    with open('data/i2w' + '.pkl', 'rb') as f:
        i2w = pickle.load(f)

    # read word pairs from disk
    dataset = []
    with open("data/DATA.txt", "r") as f:
        content = f.readlines()
    for line in content:
        dataset.append(map(int, line[:-1].split(" ")))

    # make unigram table for negative sampling
    word_freq = np.ones((len(i2w),), dtype=np.float64)
    for t in range(len(i2w)):
        word_freq[t] = a_counter[i2w[t]] ** 0.75
    word_freq = word_freq / np.sum(word_freq)

    # give a fixed size to unigram table
    size_unigram = 0
    for freq in word_freq:
        size_unigram += int(freq*100e6)

    # start to fill in unigram table with words
    unigram_table = np.zeros((size_unigram,), dtype=np.int64) - 1
    index = 0
    for i in range(len(i2w)):
        toBeAdded = int(word_freq[i] * 100e6)
        unigram_table[index: index + toBeAdded] = i
        index += toBeAdded

    # unigram quality check
    assert -1 not in unigram_table
    error = 0
    counter_check = collections.Counter(unigram_table)
    for i in counter_check:
        error += abs(counter_check[i]/float(unigram_table.shape[0]) - word_freq[i])
    print("Unigram table generated with error %f" % (error/ unigram_table.shape[0]))

    return np.array(dataset, dtype=np.int64), a_counter, w2i, i2w, unigram_table


def select_negative_samples(unigram, nb_samples, target):
    """
    Select a number of negative samples from unigram table
    :param unigram: unigram table
    :param nb_samples: number of samples to be returned
    :param target: true class to be excluded
    :return: negative samples
    """
    res = []
    while len(res) < nb_samples:
        neg_sample = random.choice(unigram)
        if neg_sample == target:  # skip if same as target class
            continue
        res.append(neg_sample)
    assert target not in res
    return res


def save_vectors(vectors):
    """
    Save embedding vectors
    :param vectors:
    :return: None
    """
    np.save("data/emb.npy", vectors)


def load_vectors(filename="data/emb.npy"):
    """
    Load embedding vectors
    :param filename: directory of .npy file
    :return: np array, word2int, int2word
    """
    with open('data/w2i' + '.pkl', 'rb') as f:
        w2i = pickle.load(f)
    with open('data/i2w' + '.pkl', 'rb') as f:
        i2w = pickle.load(f)
    return np.load(filename), w2i, i2w


def read_analogies(_file, dictionary):
    """
    Read analogy questions
    :param _file: directory to text file
    :param dictionary: word2int
    :return: questions
    """
    questions = []
    questions_skipped = 0
    with open(_file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", _file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)
