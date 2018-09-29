import numpy as np
import itertools


CLASSES = {'R-A4': 0, 'C-AM-DIR': 30, 'R-A0': 2, 'R-A1': 3, 'AM-MNR': 4, 'R-A3': 5,
           'C-AM-MNR': 6, 'R-AM-MNR': 7, 'R-AM-TMP': 8, 'AM-PRD': 9, 'C-AM-NEG': 44,
           'R-AM-DIR': 16, 'C-AM-CAU': 11, 'R-A2': 12, 'C-AM-TMP': 13, 'AM-EXT': 15,
           'R-AM-CAU': 17, 'A1': 18, 'A0': 19, 'A3': 20, 'A2': 21, 'A5': 22, 'A4': 23,
           'R-AM-EXT': 24, 'C-AM-PNC': 25, 'AM-DIR': 26, 'AM-DIS': 27, 'AM-TMP': 28,
           'AM-REC': 29, 'AA': 1, 'C-AM-DIS': 31, 'AM-PNC': 32, 'AM-LOC': 33, 'AM-TM': 34,
           'AM': 35, 'R-AM-LOC': 36, 'AM-ADV': 51, 'AM-MOD': 38, 'AM-CAU': 39, 'C-AM-LOC': 40,
           '_': 41, 'C-R-AM-TMP': 42, 'R-AM-ADV': 43, 'AM-PRT': 10, 'C-A3': 45, 'C-A2': 46, 'C-A1': 47,
           'C-A0': 48, 'R-AA': 49, 'C-A4': 14, 'C-AM-EXT': 37, 'C-AM-ADV': 52, 'R-AM-PNC': 50, 'AM-NEG': 53}
INV_CLASSES = {v: k for k, v in CLASSES.iteritems()}
POS2INT = {'vb': 0, 'prf': 1, 'cc': 2, 'jjs': 3, 'pos': 4, 'cd': 5, "''": 6, 'vbn': 42, 'prp': 7, 'in': 8, 'wp$': 9,
           'nns': 10, 'nnp': 11, '#': 12, 'nnps': 13, 'wrb': 14, '$': 15, 'nn': 16, ')': 17, '(': 18, ',': 19, '.': 20,
           'to': 21, 'ls': 22, 'rb': 23, ':': 25, 'hyph': 45, '``': 27, 'nil': 28, 'fw': 29, 'sym': 30, 'jjr': 31,
           'jj': 32, 'wp': 33, 'rp': 34, 'dt': 35, 'md': 36, 'prp$': 37, 'vbg': 38, 'vbd': 39, 'pdt': 40, 'rbs': 41,
           'rbr': 24, 'vbp': 43, 'wdt': 44, 'uh': 26, 'vbz': 46, 'ex': 47}
DEP2INT = {'prd': 0, 'posthon': 1, 'vc': 2, 'suffix': 3, 'gap-nmod': 4, 'mnr': 5, 'prn': 6, 'gap-oprd': 37,
           'hmod': 8, 'prt': 9, 'im': 10, 'gap-prp': 20, 'conj': 13, 'mnr-prd': 29, 'loc-tmp': 15, 'tmp': 16,
           'loc': 17, 'dtv-gap': 18, 'sub': 19, 'title': 21, 'pmod': 22, 'voc': 24, 'gap-pmod': 25, 'gap-put': 26,
           'appo': 27, 'gap-lgs': 28, 'nmod': 57, 'loc-prd': 47, 'gap-loc': 30, 'dir-prd': 31, 'bnf': 32, 'adv': 33,
           'gap-sub': 34, 'dep-gap': 35, 'put': 56, 'dir-oprd': 36, 'gap-tmp': 42, 'gap-mnr': 38, 'adv-gap': 39,
           'prd-prp': 40, 'ext-gap': 41, 'gap-obj': 7, 'loc-oprd': 12, 'lgs': 44, 'mnr-tmp': 45, 'name': 52,
           'amod-gap': 23, 'prd-tmp': 48, 'gap-vc': 49, 'gap-sbj': 50, 'obj': 51, 'extr': 46, 'dep': 53,
           'extr-gap': 54, 'loc-mnr': 55, 'sbj': 43, 'coord': 11, 'amod': 14, 'p': 59, 'ext': 60, 'hyph': 61,
           'gap-loc-prd': 62, 'dir-gap': 63, 'gap-prd': 64, 'oprd': 58, 'root': 65, 'dtv': 66, 'dir': 67, 'prp': 68}


class Sentence:
    def __init__(self, words, lemmas, nb_predicates, predicate_locations, predicate_meanings, predicate_arguments,
                 pos_tags, dep_tags,
                 predicate_window_size=1, use_pos=False, use_dep=False):
        """
        :param words: sentence in raw form
        :param lemmas: sentence in lemma form
        :param nb_predicates: nb of predicates in sentence
        :param predicate_locations: one hot vector representing locations of predicates
        :param predicate_meanings: meanings of predicates
        :param predicate_arguments: (nb_predicates, len(words)) label for each predicate
        :param predicate_window_size: size of predicate context
        """
        self.words = map(lambda k: k.lower(), words)
        self.lemmas = lemmas
        self.nb_predicates = nb_predicates
        self.predicate_locations = np.flatnonzero(predicate_locations)
        self.predicate_meanings = predicate_meanings
        self.predicate_arguments = predicate_arguments
        self.predicate_window_size = predicate_window_size
        self.pos_tags = map(lambda p: POS2INT[p.lower()], pos_tags)
        self.dep_tags = map(lambda d: DEP2INT[d.lower()], dep_tags)
        self.use_pos = use_pos
        self.use_dep = use_dep

    def predicate_context(self, pred_loc, arg_loc):
        pred_window = self.predicate_window_size
        indices = np.clip([pred_loc-pred_window, pred_loc+pred_window+1], 0, len(self.words))
        assert indices[0] >= 0 and indices[1] <= len(self.words)
        pred_context = self.words[indices[0]: indices[1]]
        if pred_loc-pred_window < 0:
            for _ in range(-pred_loc+pred_window):
                pred_context.insert(0, "__zero__")
        if pred_loc+pred_window+1 > len(self.words):
            for _ in range(pred_loc+pred_window+1-len(self.words)):
                pred_context.append("__zero__")
        assert len(pred_context) == pred_window*2+1
        return pred_context, int(indices[0] <= arg_loc < indices[1])

    def return_ds(self):
        all_ds = []
        for p in range(self.nb_predicates):
            _seq_features = []
            if len(self.predicate_arguments) == 0:
                _seq_labels = ["AM-ADV"]
            else:
                _seq_labels = self.predicate_arguments[p]
            for _w in range(len(self.words)):
                _feature = []
                _feature.extend([self.words[_w], self.lemmas[self.predicate_locations[p]]])  # argument, predicate
                _feature.extend(self.predicate_context(self.predicate_locations[p], _w)[0])  # predicate context
                _feature.append(self.predicate_context(self.predicate_locations[p], _w)[1])  # bool if inside pred context
                if self.use_pos:
                    _feature.append(self.pos_tags[_w])  # pos
                if self.use_dep:
                    _feature.append(self.dep_tags[_w])  # dependency
                _seq_features.append(_feature)

                # predicate 1, argument 1, predicate context 3, inside region 1, pos, dep
                assert len(_feature) == 4+self.predicate_window_size*2+int(self.use_pos)+int(self.use_dep)
            ds = DataSample(_seq_features, _seq_labels)
            all_ds.append(ds)
        return all_ds


class DataSample:
    """
    span based
    """
    def __init__(self, seq_features, seq_labels):
        self.seq_features = seq_features
        self.seq_labels = [CLASSES[r] for r in seq_labels]

    def __str__(self):
        print self.seq_features


"""
CONLL 2009 format
col 1: form
col 2: lemma
col 12: 1 hot predicate
col 13: predicate meaning
col 14 ...: labels
"""


def read_data(file_name="CoNLL2009-ST-English-train.txt", limit=-1, pred_win_size=1, use_pos=True, use_dep=True):
    """
    Read data for training
    :param file_name:
    :param limit: limit on train set size and test set size
    :param pred_win_size: predicate window size
    :param use_pos: use POS information
    :param use_dep: use DEP information
    :return:
    """
    data = open(file_name)
    all_samples = []
    for k, sen in itertools.groupby(data, key=lambda x: x.strip() != ""):
        sen = list(sen)
        if len(sen) == 1:
            continue
        row_length = None
        sen_arr = []
        for word in sen:
            word = word.strip("\n").split("\t")
            if row_length is None:
                row_length = len(word)
            assert len(word) == row_length
            sen_arr.append(word)
        sen_arr = np.array(sen_arr)
        pred_loc = np.array(map(lambda u: 1 if u == "Y" else 0, sen_arr[:, 12]))
        pred_meaning = map(lambda u: None if u == "_" else u, sen_arr[:, 13])
        pred_args = np.transpose(sen_arr[:, 14:])
        sentence_holder = Sentence(sen_arr[:, 1], sen_arr[:, 2], sum(pred_loc), pred_loc, pred_meaning, pred_args,
                                   sen_arr[:, 4], sen_arr[:, 10], predicate_window_size=pred_win_size,
                                   use_pos=use_pos, use_dep=use_dep)
        all_samples.extend(sentence_holder.return_ds())
        if limit != -1 and len(all_samples) > limit:
            break
    return all_samples


def read_data_by_packet(file_name="CoNLL2009-ST-English-train.txt", limit=-1,
                        pred_win_size=2, use_pos=True, use_dep=True):
    """
    Read data for testing (inference)
    :param file_name:
    :param limit:
    :param pred_win_size:
    :param use_pos:
    :param use_dep:
    :return:
    """
    data = open(file_name)
    all_samples = {}
    for k, sen in itertools.groupby(data, key=lambda x: x.strip() != ""):
        sen = list(sen)
        if len(sen) == 1:
            continue
        row_length = None
        sen_arr = []
        for word in sen:
            word = word.strip("\n").split("\t")
            if row_length is None:
                row_length = len(word)
            assert len(word) == row_length
            sen_arr.append(word)
        sen_arr = np.array(sen_arr)
        pred_loc = np.array(map(lambda u: 1 if u == "Y" else 0, sen_arr[:, 12]))
        pred_meaning = map(lambda u: None if u == "_" else u, sen_arr[:, 13])
        pred_args = np.transpose(sen_arr[:, 14:])
        sentence_holder = Sentence(sen_arr[:, 1], sen_arr[:, 2], sum(pred_loc), pred_loc, pred_meaning, pred_args,
                                   sen_arr[:, 4], sen_arr[:, 10], predicate_window_size=pred_win_size,
                                   use_pos=use_pos, use_dep=use_dep)
        all_samples[len(all_samples)] = (sen_arr[:, :14], sentence_holder.return_ds())
        assert sen_arr[:, :14].shape[0] > 0
        if limit != -1 and len(all_samples) > limit:
            break
    return all_samples


def make_batch(samples, batch_size=32):
    """
    Batch generator
    :param samples:
    :param batch_size:
    :return:
    """
    for _j in range(0, len(samples), batch_size):
        yield samples[_j: _j+batch_size]


def read_vocab(file_names=("CoNLL2009-ST-English-train.txt", "CoNLL2009-ST-English-development.txt", "test.csv")):
    """
    Read the needed vocab
    :param file_names:
    :return:
    """
    _vocab = []
    for name in file_names:
        data = open(name)
        for k, sen in itertools.groupby(data, key=lambda x: x.strip() != ""):
            sen = list(sen)
            if len(sen) == 1:
                continue
            row_length = None
            sen_arr = []
            for word in sen:
                word = word.strip("\n").split("\t")
                if row_length is None:
                    row_length = len(word)
                assert len(word) == row_length
                sen_arr.append(word)
            sen_arr = np.array(sen_arr)
            _vocab.extend(map(lambda t: t.lower(), list(sen_arr[:, 1])))
    return _vocab


def read_all_pos_dependency(file_names=("CoNLL2009-ST-English-train.txt",
                                        "CoNLL2009-ST-English-development.txt", "test.csv")):
    """
    Read all possible values of POS and DEP
    :param file_names:
    :return:
    """
    pos_set = []
    dependency_set = []
    for name in file_names:
        data = open(name)
        for k, sen in itertools.groupby(data, key=lambda x: x.strip() != ""):
            sen = list(sen)
            if len(sen) == 1:
                continue
            row_length = None
            sen_arr = []
            for word in sen:
                word = word.strip("\n").split("\t")
                if row_length is None:
                    row_length = len(word)
                assert len(word) == row_length
                sen_arr.append(word)
            sen_arr = np.array(sen_arr)
            pos_set.extend(map(lambda t: t.lower(), list(sen_arr[:, 4])))
            dependency_set.extend(map(lambda t: t.lower(), list(sen_arr[:, 10])))
        pos_set = set(pos_set)
    dependency_set = set(dependency_set)
    pos2int = {}
    for i in pos_set:
        pos2int[i] = len(pos2int)
    dep2int = {}
    for j in dependency_set:
        dep2int[j] = len(dep2int)
    print pos2int
    print dep2int
    return pos_set, dependency_set


def read_all_possible_labels(file_names=("CoNLL2009-ST-English-train.txt",)):
    """
    Read all possible values of SRL tags
    :param file_names:
    :return:
    """
    sen_arr = []

    for name in file_names:
        data = open(name)
        for k, sen in itertools.groupby(data, key=lambda x: x.strip() != ""):
            sen = list(sen)
            if len(sen) == 1:
                continue
            for word in sen:
                word = word.strip("\n").split("\t")
                sen_arr.extend(word[14:])
    sen_arr = set(sen_arr)
    res = {}
    for i in sen_arr:
        res[i] = len(res)
    return res


def pad_sequence(seq, max_length, feature_len, pad_to_beginning=False):
    """
    Pad a sequence with zeros till the max length
    :param seq: [w1, w2, w3 ...]
    :param max_length: pad to this length
    :param feature_len: shape of the padded value
    :param pad_to_beginning: pad to the beginning of sequence instead of the end
    :return:
    """
    for _ in range(max_length - len(seq)):
        if not pad_to_beginning:
            if feature_len == 1:
                seq.append(0)
            else:
                seq.append(np.zeros(shape=(feature_len,)))
        else:
            if feature_len == 1:
                seq.insert(0, 0)
            else:
                seq.insert(0, np.zeros(shape=(feature_len,)))
    for s in seq:
        if isinstance(s, list):
            assert len(s) == feature_len
    return seq


def create_mask(lengths, max_length, true_labels):
    res = []
    for d in range(lengths.shape[0]):
        an_array = list(np.ones(shape=(lengths[d],)))
        for _ in range(max_length - len(an_array)):
            an_array.append(0)
        res.append(an_array)
    res = np.array(res)
    return res * np.array(true_labels != CLASSES["_"], dtype=np.int32)


def report_accuracy(lengths, max_length, true_labels, predictions):
    ind = []
    for p in range(true_labels.shape[0]):
        unique_count = np.unique(true_labels[p][:lengths[p]])
        if list(unique_count) == [CLASSES["_"]]:
            ind.append(False)
        else:
            ind.append(True)
    to_be_divided = []
    for q in range(true_labels.shape[0]):
        worth_element = 0
        to_checked = list(true_labels[q][:lengths[q]])
        for r in to_checked:
            if r != CLASSES["_"]:
                worth_element += 1
        to_be_divided.append(worth_element)

    to_be_divided = np.array(to_be_divided)
    true_labels = true_labels[ind]
    lengths = lengths[ind]
    predictions = predictions[ind]
    to_be_divided = to_be_divided[ind]
    mask = create_mask(lengths, max_length, true_labels)
    acc = np.array(true_labels == predictions, dtype=np.int32) * mask
    return np.mean(np.sum(acc, axis=1) / to_be_divided)


def report_TP_FP_FN(lengths, true_labels, predictions):
    """
    Compute TP FP FN
    :param lengths: lengths of each sequence
    :param true_labels: true tags
    :param predictions: predicted tags
    :return:
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    skip = 0
    un_eval_tokens = []
    total = 0
    for g in range(predictions.shape[0]):
        gold = true_labels[g, :lengths[g]]
        pred = predictions[g, :lengths[g]]
        total += gold.shape[0]
        for h in range(gold.shape[0]):
            if gold[h] == pred[h] and gold[h] != CLASSES["_"] and gold[h] != 101:
                true_positives += 1
            elif gold[h] != pred[h] and pred[h] != CLASSES["_"]:
                false_positives += 1
            elif gold[h] != pred[h] and pred[h] == CLASSES["_"]:
                false_negatives += 1
            else:
                skip += 1
                un_eval_tokens.append(gold[h])
    return true_positives, false_positives, false_negatives, skip


def report_precision_recall_f1(stats):
    """
    Compute P R F1 based on TP FP FN
    :param stats:
    :return:
    """
    tp, fp, fn, sk = stats
    try:
        precision = float(tp) / (tp+fp)
        recall = float(tp) / (tp+fn)
        f1 = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        return 0, 0, 0
    else:
        return precision, recall, f1



if __name__ == "__main__":
    # golds = np.array([[1, 41, 41, 3, 41, 41, 41, 0, 41, 2, 0]])
    # preds = np.array([[1, 41, 41, 0, 41, 3, 41, 41, 41, 2, 0]])
    # length = np.array([10])
    # print report_TP_FP_FN(length, golds, preds)
    # read_data_by_packet(file_name="CoNLL2009-ST-English-development.txt", limit=10)
    read_data(limit=10)