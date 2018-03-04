import numpy as np
import os
import os.path
from keras.datasets import cifar10, cifar100
from keras.models import load_model
from sklearn.metrics import accuracy_score


def plot_model_history(model_history, show=False):
    if not show:
        # Force matplotlib to not use any Xwindows backend.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    if not show:
        plt.savefig('fig.png')
    else:
        plt.show()


def plot_model_history_by_list(acc, val_acc, loss, val_loss, show=False):
    if not show:
        # Force matplotlib to not use any Xwindows backend.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(acc) + 1), acc)
    axs[0].plot(range(1, len(val_acc) + 1), val_acc)
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(acc) + 1), len(acc) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(loss) + 1), loss)
    axs[1].plot(range(1, len(val_loss) + 1), val_loss)
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(loss) + 1), len(loss) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    if not show:
        plt.savefig('fig.png')
    else:
        plt.show()


def read_data_cifar10():
    (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()

    train_features = train_features.astype('float32') / 255
    test_features = test_features.astype('float32') / 255

    return train_features, train_labels.reshape((train_labels.shape[0],)), \
           test_features, test_labels.reshape((test_labels.shape[0],))


def read_data_cifar100():
    (train_features, train_labels), (test_features, test_labels) = cifar100.load_data()

    train_features = train_features.astype('float32') / 255
    test_features = test_features.astype('float32') / 255

    return train_features, train_labels.reshape((train_labels.shape[0],)), \
           test_features, test_labels.reshape((test_labels.shape[0],))


def predict_using_snapshot_ensembles(x, y, acc_dict, dir_to_models="models", m="all"):
    model_names = sorted([f for f in os.listdir(dir_to_models)
                          if os.path.isfile(os.path.join(dir_to_models, f))],
                         key=lambda x: int(x[5: x.find(".")]), reverse=True)
    models = [load_model("%s/%s" % (dir_to_models, i)) for i in model_names]
    predictions = [model.predict(x) for model in models]
    for i in range(2, len(models)+1):
        if m == "all" or i == m:
            y_pred = np.sum(predictions[:i], axis=0)/float(i)
            acc = accuracy_score(y, np.argmax(y_pred, axis=1))
            acc_dict[len(predictions[:i])].append(acc)

