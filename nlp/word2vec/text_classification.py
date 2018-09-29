import data_preprocessing as dp
import numpy as np
import tensorflow as tf
import re
import os
import h5py
import sys
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.externals

classes = {'ANIMALS': 32, 'SPORT_AND_RECREATION': 14, 'POLITICS_AND_GOVERNMENT': 8, 'METEOROLOGY': 19,
           'HEALTH_AND_MEDICINE': 0, 'RELIGION_MYSTICISM_AND_MYTHOLOGY': 27, 'GAMES_AND_VIDEO_GAMES': 2,
           'CULTURE_AND_SOCIETY': 6, 'NUMISMATICS_AND_CURRENCIES': 21, 'TRANSPORT_AND_TRAVEL': 30,
           'GEOLOGY_AND_GEOPHYSICS': 13, 'GEOGRAPHY_AND_PLACES': 26, 'COMPUTING': 12, 'TEXTILE_AND_CLOTHING': 23,
           'LAW_AND_CRIME': 10, 'ART_ARCHITECTURE_AND_ARCHAEOLOGY': 33, 'PHYSICS_AND_ASTRONOMY': 24,
           'WARFARE_AND_DEFENSE': 4, 'MEDIA': 25, 'ROYALTY_AND_NOBILITY': 3, 'BUSINESS_ECONOMICS_AND_FINANCE': 29,
           'LANGUAGE_AND_LINGUISTICS': 15, 'MUSIC': 9, 'MATHEMATICS': 5, 'FARMING': 16, 'HISTORY': 11, 'BIOLOGY': 20,
           'LITERATURE_AND_THEATRE': 31, 'HERALDRY_HONORS_AND_VEXILLOLOGY': 1, 'CHEMISTRY_AND_MINERALOGY': 28,
           'PHILOSOPHY_AND_PSYCHOLOGY': 22, 'EDUCATION': 17, 'ENGINEERING_AND_TECHNOLOGY': 18, 'FOOD_AND_DRINK': 7}
inverse_classes = dict(zip(classes.values(), classes.keys()))


def read_data(sess, w2i, directory="dataset/DATA/TRAIN", domain_limit=-1):
    """
    Read the dataset into a list of preprocessed words
    :param sess: tf session to run
    :param w2i: word2int
    :param directory: dir of training data
    :param domain_limit: limitation on number of text file to read per domain
    :return: inputs, outputs
    """
    folders = [x[0] for x in os.walk(directory) if len(x[0].split("/")) == 4]
    inputs = []
    outputs = []
    for domain in folders:
        all_files = [os.path.join(domain, f) for f in os.listdir(domain) if os.path.isfile(os.path.join(domain, f))]
        for i in range(len(all_files))[:domain_limit]:
            with open(all_files[i], 'r') as content_file:
                content = content_file.read()
            new_vocab = np.array([w2i.get(token, w2i["UNK"])
                                  for token in map(dp.preprocess_word, re.split("[ \\n\\t]", content))
                                  if token != "" and token != " "])
            res = sess.run([avg_emb], feed_dict={indices: new_vocab, shape: np.float64(len(new_vocab))})

            # skip if nan or inf is detected
            if True in np.isnan(res[0]):
                continue
            if True in np.isinf(res[0]):
                continue

            # if okay add to dataset
            inputs.append(res[0])
            outputs.append(classes[domain.split("/")[-1]])
    return np.array(inputs), np.array(outputs)


def read_testdata(sess, w2i, directory="dataset/TEST"):
    inputs = []
    names = []
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for i in range(len(all_files)):
        with open(all_files[i], 'r') as content_file:
            content = content_file.read()
        new_vocab = np.array([w2i.get(token, w2i["UNK"])
                              for token in map(dp.preprocess_word, re.split("[ \\n\\t]", content))
                              if token != "" and token != " "])
        res = sess.run([avg_emb], feed_dict={indices: new_vocab, shape: np.float64(len(new_vocab))})

        # skip if nan or inf is detected
        if True in np.isnan(res[0]):
            continue
        if True in np.isinf(res[0]):
            continue

        # if okay add to dataset
        inputs.append(res[0])
        names.append(all_files[i].split("/")[-1])
    return np.array(inputs), names


"""
TF graph for embedding lookup
"""
embeddings, word2int, int2word = dp.load_vectors()
graph = tf.Graph()
with graph.as_default():
    indices = tf.placeholder(tf.int64, shape=[None])
    shape = tf.placeholder(tf.float32)
    input_emb = tf.nn.embedding_lookup(embeddings, indices)
    avg_emb = tf.reduce_sum(input_emb, 0) / shape

"""
To read data file, uncomment this below section for the first time you run
"""
# with tf.Session(graph=graph) as session:
#     x_train2, y_train2 = read_data(session, word2int, domain_limit=100)
#     x_test2, y_test2 = read_data(session, word2int, directory="dataset/DATA/DEV", domain_limit=100)
#
# with h5py.File('data/data_classify.h5', 'w') as hf:
#     hf.create_dataset("xtrain", data=x_train2)
#     hf.create_dataset("ytrain", data=y_train2)
#     hf.create_dataset("xtest", data=x_test2)
#     hf.create_dataset("ytest", data=y_test2)

"""
To load data file
"""
with h5py.File('data/data_classify.h5', 'r') as hf:
    x_train = hf['xtrain'][:]
    x_test = hf['xtest'][:]
    y_train = hf['ytrain'][:]
    y_test = hf['ytest'][:]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print("Read data done")

"""
To tune the classifier
"""
# tuned_parameters = {"C": [0.1, 1, 10], "solver": ["saga", "liblinear"], "penalty": ["l1", "l2"]}
# clf = sklearn.model_selection.GridSearchCV(sklearn.linear_model.LogisticRegression(max_iter=1000), tuned_parameters,
#                                            scoring="f1_weighted", n_jobs=-1)
# clf = clf.fit(x_train, y_train)
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)

"""
To train and save the optimal classifier
"""
model = sklearn.linear_model.LogisticRegression(C=100, n_jobs=-1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(sklearn.metrics.accuracy_score(y_test, y_pred))
print(sklearn.metrics.precision_score(y_test, y_pred, average="weighted"))
print(sklearn.metrics.recall_score(y_test, y_pred, average="weighted"))
print(sklearn.metrics.f1_score(y_test, y_pred, average="weighted"))
sklearn.externals.joblib.dump(model, "data/text_classifier.pkl")

"""
To predict test set and write predictions
"""
with tf.Session(graph=graph) as session:
    x_pred, filenames = read_testdata(session, word2int)
    print("Reading test data done")
y_pred = model.predict(x_pred)
sys.stdout = open("pred.txt", "w")
for i in range(x_pred.shape[0]):
    print("%s\t%s\n" % (filenames[i][:-4], inverse_classes[y_pred[i]]))