from keras.utils import to_categorical
from scipy.ndimage import zoom
from skimage.transform import resize
from os import listdir, environ
from keras.models import Model
from keras.preprocessing import image

import sys
import skimage.io
import skimage.transform
import numpy as np
import random
import keras.applications

CLASSES2INT = {'Mooping floor': 0, 'Knitting': 2, 'Ironing clothes': 1,
               'Hand washing clothes': 4, 'Polishing forniture': 5,
               'Vacuuming floor': 8, 'Polishing shoes': 7, 'Cleaning shoes': 9,
               'Cleaning windows': 6, 'Wrapping presents': 3}


def split_videos():
    all_videos = listdir("extracted_videos/")
    all_video_keys = ["_".join(u.split("_")[:-1]) for u in all_videos]
    a_dict = {u: "_".join(u.split("_")[:-1]) for u in all_videos}
    train_video_keys = []
    test_video_keys = []
    train_videos = []
    test_videos = []
    for c in CLASSES2INT:
        relevant_videos = [v for v in all_video_keys if c in v]
        train_video_keys.extend(relevant_videos[:int(len(relevant_videos) * 0.7)])
        test_video_keys.extend(relevant_videos[int(len(relevant_videos) * 0.7):])
    for g in all_videos:
        if a_dict[g] in train_video_keys:
            train_videos.append(g)
        elif a_dict[g] in test_video_keys:
            test_videos.append(g)
    print(len(train_videos), len(test_videos))
    print(len(train_video_keys), len(test_video_keys))
    sys.stdout = open("split_info/train_videos.txt", "w")
    for f in train_videos:
        print(f)
    sys.stdout = open("split_info/test_videos.txt", "w")
    for f in test_videos:
        print(f)
    sys.stdout = sys.__stdout__


def split_frames():
    all_frames = listdir("/home/ditto/vision/frames")
    video_names = set(["_".join(v.split("_")[:-2]) for v in all_frames])
    classes2int = {'Mooping floor': 0, 'Knitting': 2, 'Ironing clothes': 1,
                   'Hand washing clothes': 4, 'Polishing forniture': 5,
                   'Vacuuming floor': 8, 'Polishing shoes': 7, 'Cleaning shoes': 9,
                   'Cleaning windows': 6, 'Wrapping presents': 3}
    train_videos = []
    test_videos = []
    for c in classes2int:
        relevant_videos = [v for v in video_names if c in v]
        train_videos.extend(relevant_videos[:int(len(relevant_videos) * 0.7)])
        test_videos.extend(relevant_videos[int(len(relevant_videos) * 0.7):])
    print(len(train_videos), len(test_videos))

    train_frames = []
    test_frames = []
    for f in all_frames:
        key = "_".join(f.split("_")[:2])
        if key in train_videos:
            train_frames.append(f)
        elif key in test_videos:
            test_frames.append(f)
    print(len(train_frames), len(test_frames))

    sort_key = lambda x: ("_".join(x.split("_")[:-1]), int(x.split("_")[-1].split(".")[0]))
    train_frames = sorted(train_frames, key=sort_key)
    test_frames = sorted(test_frames, key=sort_key)

    sys.stdout = open("split_info/train_frames.txt", "w")
    for f in train_frames:
        print(f)
    sys.stdout = open("split_info/test_frames.txt", "w")
    for f in test_frames:
        print(f)
    sys.stdout = open("split_info/train_videos.txt", "w")
    for f in train_videos:
        print(f)
    sys.stdout = open("split_info/test_videos.txt", "w")
    for f in test_videos:
        print(f)
    sys.stdout = sys.__stdout__
    assert check_split_files() is True
    print("Checking is done.")


def check_split_files(train_file="split_info/train_frames.txt", test_file="split_info/test_frames.txt"):
    train_videos = []
    test_videos = []
    with open(train_file) as f:
        data = f.readlines()
    for line in data:
        line = line.rstrip()
        train_videos.append("_".join(line.split("_")[:-1]))

    with open(test_file) as f:
        data = f.readlines()
    for line in data:
        line = line.rstrip()
        test_videos.append("_".join(line.split("_")[:-1]))

    for g in test_videos:
        if g in train_videos:
            return False
    return True


def produce_image(frame_dir, nb_frames_per_seq, data_aug=True):
    print("processing %s" % frame_dir)
    frame_names = sorted(listdir("frames/%s" % frame_dir),
                         key=lambda p: int(p[:-4].split("img")[1]))[:nb_frames_per_seq]
    rand_frame_names = random.sample(frame_names, min(10, len(frame_names)))
    cond_check = [int(n[:-4].split("img")[1]) for n in frame_names]
    assert cond_check == list(range(1, max(cond_check) + 1))
    cap = np.zeros(shape=(len(rand_frame_names), 240, 320, 3))
    current_idx = 0
    for frame_name in rand_frame_names:
        frame = skimage.io.imread("frames/%s/%s" % (frame_dir, frame_name))
        if data_aug:
            frame = skimage.transform.resize(frame, (240, 320), preserve_range=True)
        else:
            frame = skimage.transform.resize(frame, (224, 224), preserve_range=True)
        cap[current_idx] = frame
        current_idx += 1
    assert np.sum(cap) != 0.0
    cap = oversample(cap, [224, 224])
    cap = preprocess_input(cap)[random.sample(range(cap.shape[0]), 10), :]
    assert cap.shape[0] == 10
    return cap, np.ones(shape=(10,)) * CLASSES2INT[frame_dir.split("_")[0]]


def read_data(process_train=False, process_test=False, limit_size=200, test_limit_size=100, nb_frames_per_seq=200):
    """
    Read data fro single frame CNN
    :param process_train:
    :param process_test:
    :param limit_size:
    :param test_limit_size:
    :param nb_frames_per_seq:
    :return:
    """
    train_seqs = []
    test_seqs = []
    train_labels = []
    test_labels = []
    skipped_videos = 0
    with open("split_info/train_videos.txt") as f:
        train_videos = [l.rstrip() for l in f.readlines()]
    with open("split_info/test_videos.txt") as f:
        test_videos = [l.rstrip() for l in f.readlines()]
    all_frame_dirs = listdir("frames/")

    print("Reading a total of %d, including %d train and %d test" % (len(all_frame_dirs),
                                                                     len(train_videos[:limit_size]),
                                                                     len(test_videos[:test_limit_size])))

    for frame_dir in all_frame_dirs:
        if len(listdir("frames/%s" % frame_dir)) > 0:
            if frame_dir in train_videos[:limit_size] and process_train:
                features, labels = produce_image(frame_dir, nb_frames_per_seq)
                train_seqs.append(features)
                train_labels.append(labels)
                print("  done %d/%d train videos, %d skipped" % (len(train_labels), len(all_frame_dirs),
                                                                 skipped_videos))
            elif frame_dir in test_videos[:test_limit_size] and process_test:
                features, labels = produce_image(frame_dir, nb_frames_per_seq)
                test_seqs.append(features)
                test_labels.append(labels)
                print("  done %d/%d test videos, %d skipped" % (len(test_labels), len(all_frame_dirs), skipped_videos))
        else:
            skipped_videos += 1

    train_seqs = np.reshape(np.array(train_seqs), (-1, 224, 224, 3))
    test_seqs = np.reshape(np.array(test_seqs), (-1, 224, 224, 3))
    train_labels = to_categorical(np.reshape(np.array(train_labels), (-1, 1)), num_classes=10)
    test_labels = to_categorical(np.reshape(np.array(test_labels), (-1, 1)), num_classes=10)
    print(train_seqs.shape, test_seqs.shape, train_labels.shape, test_labels.shape)
    if process_train:
        np.save("loaded_data/x_train_single-%d.npy" % nb_frames_per_seq, train_seqs)
        np.save("loaded_data/y_train_single-%d.npy" % nb_frames_per_seq, train_labels)
    if process_test:
        np.save("loaded_data/x_test_single-%d.npy" % nb_frames_per_seq, test_seqs)
        np.save("loaded_data/y_test_single-%d.npy" % nb_frames_per_seq, test_labels)


def read_data_lstm(process_train=False, process_test=False, limit_size=200, test_limit_size=100, nb_frames_per_seq=25):
    """
    Read data for LRCN
    :param process_train:
    :param process_test:
    :param limit_size:
    :param test_limit_size:
    :param nb_frames_per_seq:
    :return:
    """

    train_seqs = []
    test_seqs = []
    train_labels = []
    test_labels = []
    train_lens = []
    test_lens = []
    skipped_videos = 0
    with open("split_info/train_videos.txt") as f:
        train_videos = [l.rstrip() for l in f.readlines()]
    with open("split_info/test_videos.txt") as f:
        test_videos = [l.rstrip() for l in f.readlines()]
    all_frame_dirs = listdir("frames/")

    print("Reading a total of %d, including %d train and %d test" % (len(all_frame_dirs),
                                                                     len(train_videos[:limit_size]),
                                                                     len(test_videos[:test_limit_size])))

    for frame_dir in all_frame_dirs:
        if len(listdir("frames/%s" % frame_dir)) > 0:
            if frame_dir in train_videos[:limit_size] and process_train:
                features, labels, seq_len = produce_bottleneck_features(None, frame_dir, nb_frames_per_seq,
                                                                        if_vgg_features=False)
                train_seqs.append(features)
                train_labels.append(labels)
                train_lens.append(seq_len)
                print("  done %d/%d train videos, %d skipped" % (len(train_labels), len(all_frame_dirs),
                                                                 skipped_videos))
            elif frame_dir in test_videos[:test_limit_size] and process_test:
                features, labels, seq_len = produce_bottleneck_features(None, frame_dir, nb_frames_per_seq,
                                                                        if_vgg_features=False)
                test_seqs.append(features)
                test_labels.append(labels)
                test_lens.append(seq_len)
                print("  done %d/%d test videos, %d skipped" % (len(test_labels), len(all_frame_dirs), skipped_videos))
        else:
            skipped_videos += 1

    train_seqs = np.array(train_seqs)
    test_seqs = np.array(test_seqs)
    train_lens = np.array(train_lens)
    test_lens = np.array(test_lens)
    train_labels = to_categorical(np.array(train_labels), num_classes=10)
    test_labels = to_categorical(np.array(test_labels), num_classes=10)
    print(train_seqs.shape, test_seqs.shape, train_labels.shape, test_labels.shape, train_lens.shape, test_lens.shape)
    if process_train:
        np.save("loaded_data/x_train_lstm-%d.npy" % nb_frames_per_seq, train_seqs)
        np.save("loaded_data/y_train_lstm-%d.npy" % nb_frames_per_seq, train_labels)
        np.save("loaded_data/len_train_lstm-%d.npy" % nb_frames_per_seq, train_lens)
    if process_test:
        np.save("loaded_data/x_test_lstm-%d.npy" % nb_frames_per_seq, test_seqs)
        np.save("loaded_data/y_test_lstm-%d.npy" % nb_frames_per_seq, test_labels)
        np.save("loaded_data/len_test_lstm-%d.npy" % nb_frames_per_seq, test_lens)


def produce_bottleneck_features(model, frame_dir, nb_frames_per_seq, if_vgg_features=True, model_name="vgg"):
    print("processing %s" % frame_dir)
    frame_names = sorted(listdir("frames/%s" % frame_dir),
                         key=lambda p: int(p[:-4].split("img")[1]))[:nb_frames_per_seq]
    cond_check = [int(n[:-4].split("img")[1]) for n in frame_names]
    assert cond_check == list(range(1, max(cond_check) + 1))

    if model_name == "vgg":
        cap = np.zeros(shape=(nb_frames_per_seq, 224, 224, 3))
        current_idx = 0
        for frame_name in frame_names:
            frame = image.load_img("frames/%s/%s" % (frame_dir, frame_name), target_size=(224, 224))
            frame = image.img_to_array(frame)
            cap[current_idx] = frame
            current_idx += 1
        assert np.sum(cap) != 0.0
        cap = keras.applications.vgg16.preprocess_input(cap)
    elif model_name == "inception":
        cap = np.zeros(shape=(nb_frames_per_seq, 299, 299, 3))
        current_idx = 0
        for frame_name in frame_names:
            frame = image.load_img("frames/%s/%s" % (frame_dir, frame_name), target_size=(299, 299))
            frame = image.img_to_array(frame)
            cap[current_idx] = frame
            current_idx += 1
        assert np.sum(cap) != 0.0
        cap = keras.applications.inception_v3.preprocess_input(cap)

    if if_vgg_features:
        return model.predict(cap, batch_size=32), CLASSES2INT[frame_dir.split("_")[0]], len(frame_names), model_name
    else:
        return cap, CLASSES2INT[frame_dir.split("_")[0]], len(frame_names)


def read_mask_and_bbox(video_name, nb_frames_per_seq):
    mask = np.load("output_coco/mask/%s.json.npy" % video_name)
    if mask.shape[0] > 0:
        mask = mask[:nb_frames_per_seq, :]
    else:
        return np.zeros(shape=(nb_frames_per_seq, 81)), np.zeros(shape=(nb_frames_per_seq, 81))
    bbox = np.load("output_coco/bbox/%s.json.npy" % video_name)[:nb_frames_per_seq, :]
    prob = np.load("output_coco/prob/%s.json.npy" % video_name)[:nb_frames_per_seq, :]
    mask = np.pad(mask, ((0, nb_frames_per_seq - mask.shape[0]), (0, 0)), mode="constant")
    bbox = np.pad(bbox, ((0, nb_frames_per_seq - bbox.shape[0]), (0, 0)), mode="constant")
    prob = np.pad(prob, ((0, nb_frames_per_seq - prob.shape[0]), (0, 0)), mode="constant")
    return mask, bbox, prob


def read_data_lstm_vgg_bottleneck(train_only=True, limit_size=200, test_limit_size=100, nb_frames_per_seq=50,
                                  model_name="vgg"):
    """
    Read data for LSTM_VGG16
    :param train_only:
    :param limit_size:
    :param test_limit_size:
    :param nb_frames_per_seq:
    :param model_name:
    :return:
    """
    if model_name == "inception":
        base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    elif model_name == "vgg":
        base_model = keras.applications.VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    train_seqs = []
    test_seqs = []
    train_labels = []
    test_labels = []
    train_lens = []
    test_lens = []
    train_masks = []
    test_masks = []
    train_bboxes = []
    test_bboxes = []
    train_prob = []
    test_prob = []
    skipped_videos = 0
    with open("split_info/train_videos.txt") as f:
        train_videos = [l.rstrip() for l in f.readlines()]
    with open("split_info/test_videos.txt") as f:
        test_videos = [l.rstrip() for l in f.readlines()]
    all_frame_dirs = listdir("frames/")

    print("Reading a total of %d, including %d train and %d test" % (len(all_frame_dirs),
                                                                     len(train_videos[:limit_size]),
                                                                     len(test_videos[:test_limit_size])))

    for frame_dir in all_frame_dirs:
        if len(listdir("frames/%s" % frame_dir)) > 0:
            if frame_dir in train_videos[:limit_size]:
                features, labels, seq_len, check = produce_bottleneck_features(model, frame_dir, nb_frames_per_seq,
                                                                               model_name=model_name)
                assert check == model_name
                mask, bbox, prob = read_mask_and_bbox(frame_dir, nb_frames_per_seq=nb_frames_per_seq)
                train_seqs.append(features)
                train_labels.append(labels)
                train_lens.append(seq_len)
                train_masks.append(mask)
                train_bboxes.append(bbox)
                train_prob.append(prob)
                print("  done %d/%d train videos, %d skipped" % (len(train_labels), len(all_frame_dirs),
                                                                 skipped_videos))
            elif frame_dir in test_videos[:test_limit_size] and not train_only:
                features, labels, seq_len, check = produce_bottleneck_features(model, frame_dir, nb_frames_per_seq,
                                                                               model_name=model_name)
                assert check == model_name
                mask, bbox, prob = read_mask_and_bbox(frame_dir, nb_frames_per_seq=nb_frames_per_seq)
                test_seqs.append(features)
                test_labels.append(labels)
                test_lens.append(seq_len)
                test_masks.append(mask)
                test_bboxes.append(bbox)
                test_prob.append(prob)
                print("  done %d/%d test videos, %d skipped" % (len(test_labels), len(all_frame_dirs), skipped_videos))
        else:
            skipped_videos += 1

    train_seqs = np.array(train_seqs)
    test_seqs = np.array(test_seqs)
    train_lens = np.array(train_lens)
    test_lens = np.array(test_lens)
    train_labels = to_categorical(np.array(train_labels), num_classes=10)
    test_labels = to_categorical(np.array(test_labels), num_classes=10)
    train_masks = np.array(train_masks)
    test_masks = np.array(test_masks)
    train_bboxes = np.array(train_bboxes)
    test_bboxes = np.array(test_bboxes)
    print(train_seqs.shape, test_seqs.shape, train_labels.shape, test_labels.shape, train_lens.shape, test_lens.shape,
          train_masks.shape, test_masks.shape, train_bboxes.shape, test_bboxes.shape)
    np.save("loaded_data/lstm/x_train_vgg_features-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")),
            train_seqs)
    np.save("loaded_data/lstm/y_train_vgg_features-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")),
            train_labels)
    np.save("loaded_data/lstm/x_test_vgg_features-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")),
            test_seqs)
    np.save("loaded_data/lstm/y_test_vgg_features-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")),
            test_labels)
    np.save("loaded_data/lstm/len_train-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), train_lens)
    np.save("loaded_data/lstm/len_test-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), test_lens)
    np.save("loaded_data/lstm/mask_train-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), train_masks)
    np.save("loaded_data/lstm/mast_test-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), test_masks)
    np.save("loaded_data/lstm/bbox_train-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), train_bboxes)
    np.save("loaded_data/lstm/bbox_test-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), test_bboxes)
    np.save("loaded_data/lstm/prob_train-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), train_prob)
    np.save("loaded_data/lstm/prob_test-%d-%d.npy" % (nb_frames_per_seq, int(model_name == "vgg")), test_prob)


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Parameters
    ----------
    images : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
        crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix - 5:ix] = crops[ix - 5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def make_batch(train_data, train_label, sequence_len, mask, bbox, prob, batch_size=32):
    """
    Batch generator that generates the same size of batch until exhausted
    :param train_data: returned by read_train_data or read_test_data
    :param train_label: labels
    :param sequence_len: length of seq
    :param mask:
    :param bbox:
    :param prob
    :param batch_size:
    :return:
    """
    assert train_label.shape[0] == train_data.shape[0]
    for i in range(0, int(train_data.shape[0] / batch_size) + 1):
        if batch_size * (i + 1) > train_data.shape[0]:
            yield train_data[train_data.shape[0] - batch_size: train_data.shape[0]], \
                  train_label[train_data.shape[0] - batch_size: train_data.shape[0]], \
                  sequence_len[train_data.shape[0] - batch_size: train_data.shape[0]], \
                  mask[train_data.shape[0] - batch_size: train_data.shape[0]], \
                  bbox[train_data.shape[0] - batch_size: train_data.shape[0]], \
                  prob[train_data.shape[0] - batch_size: train_data.shape[0]]

        else:
            yield train_data[batch_size * i: batch_size * (i + 1)], \
                  train_label[batch_size * i: batch_size * (i + 1)], \
                  sequence_len[batch_size * i: batch_size * (i + 1)], \
                  mask[batch_size * i: batch_size * (i + 1)], \
                  bbox[batch_size * i: batch_size * (i + 1)], \
                  prob[batch_size * i: batch_size * (i + 1)]


if __name__ == '__main__':
    # split_videos()
    # x_train, x_test, y_train, y_test = read_data(limit_size=10, test_limit_size=5)
    # print(y_train)
    # print(y_test)
    # print(x_train.dtype)
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    read_data_lstm_vgg_bottleneck(train_only=False, limit_size=-1, test_limit_size=-1, nb_frames_per_seq=25)
    read_data_lstm_vgg_bottleneck(train_only=False, limit_size=-1, test_limit_size=-1, nb_frames_per_seq=50)
    read_data_lstm_vgg_bottleneck(train_only=False, limit_size=-1, test_limit_size=-1, nb_frames_per_seq=100)

    read_data_lstm_vgg_bottleneck(train_only=False, limit_size=-1, test_limit_size=-1, nb_frames_per_seq=25,
                                  model_name="inception")
    read_data_lstm_vgg_bottleneck(train_only=False, limit_size=-1, test_limit_size=-1, nb_frames_per_seq=50,
                                  model_name="inception")
    read_data_lstm_vgg_bottleneck(train_only=False, limit_size=-1, test_limit_size=-1, nb_frames_per_seq=100,
                                  model_name="inception")

    # read_data_lstm(limit_size=-1, test_limit_size=-1, process_train=True)
    # read_data_lstm(limit_size=-1, test_limit_size=-1, process_test=True)
    # read_data(process_train=True, limit_size=-1)
    # read_data(process_test=True, test_limit_size=-1)

