import os
import numpy as np
import collections
import cv2
from preprocessing.image import center_crop

DataSets = collections.namedtuple('DataSets', ['train', 'validation', 'test'])
TrainvalSets = collections.namedtuple('DataSets', ['train', 'validation'])
TestSets = collections.namedtuple('DataSets', ['test'])
CLASSES = ['cat', 'dog']


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 seed=None):
        np.random.seed(seed)
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._one_hot = one_hot
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, input_shape, shuffle=True, augment=True, is_training=True):
        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]

        if self._index_in_epoch + batch_size > self._num_examples:
            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start
            end = self._num_examples
            images_rest_part = self._images[start:end]
            labels_rest_part = self._labels[start:end]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch_images = self._images[start:end]
            batch_labels = self._labels[start:end]

        batch_images = center_crop(batch_images, input_shape)

        # TODO: data augmentation routines

        return batch_images, batch_labels


def extract_data_sets(d, one_hot=True, is_trainval=True, set_size=None):
    filename_list = os.listdir(d)
    if set_size is not None and set_size < len(filename_list):
        filename_list = np.random.choice(filename_list, size=set_size, replace=False)
    np.random.shuffle(filename_list)

    images = np.empty((len(filename_list), 256, 256, 3), dtype=np.float32)
    labels = np.empty((len(filename_list)), dtype=np.int8)
    for i, filename in enumerate(filename_list):
        if (i + 1) % 100 == 0:
            print('Reading {} data: {}/{}'.format('trainval' if is_trainval else 'test', i + 1, len(filename_list)), end='\r')
        label = filename.split('.')[0]
        assert label in CLASSES, ('Undefined image label \'{}\', expect cat or dog: File \"{}\"'.format(label, filename))
        label = CLASSES.index(label)
        file_path = os.path.join(d, filename)

        image = cv2.imread(file_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        images[i] = image
        labels[i] = label

    if one_hot:
        labels_one_hot = np.zeros((len(filename_list), len(CLASSES)), dtype=np.int8)
        labels_one_hot[np.arange(len(filename_list)), labels] = 1
        labels = labels_one_hot

    print('\nDone.')

    return images, labels


def read_trainval_sets(d, one_hot=True, validation_size=2500, seed=None):

    train_images, train_labels = extract_data_sets(os.path.join(d, 'train'), one_hot=one_hot, is_trainval=True)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                         .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, one_hot=one_hot, seed=seed)
    validation = DataSet(validation_images, validation_labels, one_hot=one_hot, seed=seed)

    if validation_images.size == 0:
        validation = None

    return TrainvalSets(train=train, validation=validation)


def read_test_sets(d, one_hot=True, seed=None):

    test_images, test_labels = extract_data_sets(os.path.join(d, 'test1'), one_hot=one_hot, is_trainval=False)
    test = DataSet(test_images, test_labels, one_hot=one_hot, seed=seed)

    return TestSets(test=test)


def read_data_sets(d, one_hot=True, validation_size=2500, seed=None):

    trainval_set = read_trainval_sets(d, one_hot, validation_size, seed)
    test_set = read_test_sets(d, one_hot, seed)

    return DataSets(train=trainval_set.train, validation=trainval_set.validation, test=test_set.test)
