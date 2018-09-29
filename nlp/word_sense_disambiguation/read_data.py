import xml.etree.ElementTree as ET
import pandas as pd
import io
import sys


class DataSample:
    """
    Class to wrap one instance point in training data
    """
    def __init__(self, fw, bw, word_id, sense_id, word_int, possible_senses, lemma=""):
        if len(fw) == 0:
            fw = [""]
        if len(bw) == 0:
            bw = [""]
        self.fw = fw
        self.bw = bw
        self.word_id = word_id
        self.sense_id = sense_id
        self.word_int = word_int
        self.possible_senses = possible_senses
        self.lemma = lemma

    def __str__(self):
        return " ".join([x.__str__() for x in [self.fw, self.bw, self.word_id, self.word_int,
                                               self.sense_id, self.possible_senses]])


def read_x(fname="semcor.data.xml"):
    """
    Read xml file into data
    :param fname:
    :return:
    """
    tree = ET.parse(fname)
    root = tree.getroot()
    data = []
    for text in root:
        for sentence in text:
            data.append(sentence)
    return rawdata2pairs(data)


def read_y(fname="semcor.gold.key.bnids.txt"):
    """
    Load sense targets
    :param fname:
    :return: {word id: sense id}
    """
    df = pd.read_csv(fname, sep=" ", names=["id", "sense"])
    mat = df.as_matrix()
    a_dict = {}
    for i in range(mat.shape[0]):
        if mat[i, 0] in a_dict:
            print(mat[i, 0])
        a_dict[mat[i, 0]] = mat[i, 1]
    assert len(a_dict) == mat.shape[0]
    return a_dict


def write_all_lemmas(file_to_write="all_words.txt"):
    x_data = read_x()  # (sentence, index, ID)
    count = 0
    sys.stdout = open(file_to_write, "w")
    for sentence, idx, _ in x_data:
        print("%s\n" % sentence[idx])
        count += 1
    x_data = read_x("ALL.data.xml")  # (sentence, index, ID)
    sys.stdout = open(file_to_write, "w")
    for sentence, idx, _ in x_data:
        print("%s\n" % sentence[idx])
        count += 1
    sys.stdout = sys.__stdout__
    print("%d words written" % count)


def load_vectors(fname="pretrained_vectors/wiki-news-300d-1M.vec", limit=-1):
    """
    Load pretrained word vectors
    :param fname:
    :param limit:
    :return: a dict maps word -> vector
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        if limit != -1:
            count += 1
            if count > limit:
                break
    return data


def rawdata2pairs(_raw_data):
    """
    Preprocess raw data into instances
    :param _raw_data:
    :return: (sentence, target position, word id, lemma)
    """
    pairs = []
    vocab = []
    seq_len = []
    for sentence in _raw_data:
        text_sequence = [word.text.lower() for word in sentence]
        seq_len.append(len(text_sequence))
        vocab.extend(text_sequence)
        for w in range(len(sentence)):
            if sentence[w].tag == "instance":
                pairs.append((text_sequence, w, sentence[w].attrib["id"], sentence[w].attrib["lemma"].lower()))
    vocab = set(vocab)
    print("longest/shortest seq: %d %d" % (max(seq_len), min(seq_len)))
    return pairs, vocab


def build_lemma2senses(_data, senses):
    """
    :param _data: (sentence, index, ID, lemma)
    :param senses: {ID: sense_ID}
    :return: {lemma: [sense1, sense2, ...]} {lemma: word_int}
    """
    lemma2senses = {}
    for sentence, idx, word_id, lemma in _data:
        if lemma not in lemma2senses:
            lemma2senses[lemma] = [senses[word_id]]
        else:
            lemma2senses[lemma].append(senses[word_id])
        lemma2senses[lemma] = list(set(lemma2senses[lemma]))
    lemma2int = {}
    for k in lemma2senses:
        lemma2int[k] = len(lemma2int)
    return lemma2senses, lemma2int


def read_train_data(val_x, val_y, filter_one_meaning_words=False):
    """
    Build train data
    :param filter_one_meaning_words: filter words that have only one meaning
    :param val_x: merge target senses from val data
    :param val_y: merge target senses from val data
    :return: {word_int: [DS1, DS2, ...]}
    """
    x_data, _ = read_x()  # (sentence, index, ID, lemma)
    y_data = read_y()  # {ID: sense_ID}
    y_data.update(val_y)
    l2s, l2i = build_lemma2senses(x_data+val_x, y_data)
    train_data = []
    for sentence, idx, word_id, lemma in x_data:
        if filter_one_meaning_words and len(l2s[lemma]) == 1:
            continue
        ds = DataSample(sentence[:idx], sentence[idx+1:], word_id, y_data[word_id],
                        l2i[lemma], l2s[lemma])  # (fw, bw, word_id, sense_id, word_int, possible_senses)
        train_data.append(ds)
    print("Read %d samples" % len(train_data))
    return train_data, l2s, l2i


def read_test_data(train_lemma2int, train_lemma2senses, word2vec, fname="ALL.data.xml",
                    fname_target="ALL.gold.key.bnids.txt"):
    """
    Read test data
    :param train_lemma2int: to remove test samples that is unseen
    :param train_lemma2senses: to get the possible senses for each test sample
    :param word2vec: initialized word2vec (to check for safe)
    :param fname: test instance file
    :param fname_target: test target file
    :return: [(sentence, index, ID)]
    """
    x_data, _ = read_x(fname)  # (sentence, index, ID, lemma)
    x_data2 = []
    for p in x_data:
        if p[-1] in train_lemma2int:
            x_data2.append(p)
    print("Test data with size %d contains %d words" % (len(x_data2), len(set([s[0][s[1]] for s in x_data2]))))
    y_data = read_y(fname_target)
    test_data = []
    for sentence, idx, word_id, lemma in x_data2:
        assert y_data[word_id] in train_lemma2senses[lemma]
        for w in sentence:
            assert w in word2vec
        ds = DataSample(sentence[:idx], sentence[idx + 1:], word_id, y_data[word_id],
                        train_lemma2int[lemma], train_lemma2senses[lemma], lemma)
        test_data.append(ds)
    return test_data, y_data, x_data2


def make_batch(train_data, batch_size=32):
    """
    Batch generator that generates the same size of batch until exhausted
    :param train_data: returned by read_train_data or read_test_data
    :param batch_size:
    :return:
    """
    for i in range(0, len(train_data)/batch_size+1):
        if batch_size*(i+1) > len(train_data):
            yield train_data[len(train_data)-batch_size: len(train_data)]
        else:
            yield train_data[batch_size*i: batch_size*(i+1)]


def read_test_words(l2i, l2s, fname="test_data.txt"):
    """
    Read test files to perform predictions
    :param l2i: lemma2int
    :param l2s: lemma2senses
    :param fname:
    :return: [(sequence, target index, target id, lemma)]
    """
    with open(fname) as f:
        content = f.readlines()
    raw_data = []
    for sen in content:
        seq = [t.split("|")[0].lower() for t in sen.split(" ")][:-1]
        tokens = sen.split(" ")
        for i in range(len(tokens)-1):
            chars = tokens[i].split("|")
            if len(chars) == 4:
                raw_data.append((seq, i, chars[-1], chars[1].lower()))
    test_data = []
    for sentence, idx, word_id, lemma in raw_data:
        if lemma in l2i:
            ds = DataSample(sentence[:idx], sentence[idx + 1:], word_id, None,
                            l2i[lemma], l2s[lemma])  # (fw, bw, word_id, sense_id, word_int, possible_senses)
            test_data.append((ds, "in", lemma))
        else:
            test_data.append((word_id, "out"))
    return test_data
