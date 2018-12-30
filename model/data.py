import os
import glob
import random
import csv
import numpy as np
import keras.preprocessing.sequence as pp
import tensorflow as tf
import itertools
#from itertools import izip_longest

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)


def file_name(abs_path):
    return os.path.basename(abs_path).replace('.csv', '')


def format_row(row, shuffle=True, multilabel=False):
    y, raw_x = row
    x = np.array([int(v) for v in raw_x.split()])
    if shuffle:
        np.random.shuffle(x)
    
    if multilabel:
        return [y, x, len(x)]
    else:
        return [int(y), x, len(x)]

def format_row_bidirectional(row, shuffle=True, multilabel=False):
    y, raw_x = row
    x = np.array([int(v) for v in raw_x.split()])
    if shuffle:
        np.random.shuffle(x)
    x_rev = x[::-1]

    if multilabel:
        return [y, x, x_rev, len(x)]
    else:
        return [int(y), x, x_rev, len(x)]


class Dataset(object):
    def __init__(self, path):
        self.path = path
        files = glob.glob(path + '/*.csv')
        self.collections = {file_name(file): file for file in files}

    def rows(self, collection_name, num_epochs=None):
        if collection_name not in self.collections:
            raise ValueError(
                'Collection not found: {}'.format(collection_name)
            )
        epoch = 0
        while True:
            #with open(self.collections[collection_name], 'rb', newline='') as f:
            with open(self.collections[collection_name], 'r') as f:
                r = csv.reader(f)
                for row in r:
                    yield row
            epoch += 1
            if num_epochs and (epoch >= num_epochs):
                raise StopIteration

    def _batch_iter(self, collection_name, batch_size, num_epochs):
        gen = [self.rows(collection_name, num_epochs)] * batch_size
        return itertools.zip_longest(fillvalue=None, *gen)
        #return izip_longest(fillvalue=None, *gen)

    def batches(self, collection_name, batch_size, num_epochs=None, shuffle=True, max_len=None, multilabel=False):
        for batch in self._batch_iter(collection_name, batch_size, num_epochs):
            data = [format_row(row, shuffle=shuffle, multilabel=multilabel) for row in batch if row]
            y, x, seq_lengths = zip(*data)
            if not max_len is None:
                x = pp.pad_sequences(x, maxlen=max_len, padding='post')
            else:
                x = pp.pad_sequences(x, padding='post')
            yield np.array(y), x, np.array(seq_lengths)

    def batches_bidirectional(self, collection_name, batch_size, num_epochs=None, shuffle=True, max_len=None, multilabel=False):
        for batch in self._batch_iter(collection_name, batch_size, num_epochs):
            data = [format_row_bidirectional(row, shuffle=shuffle, multilabel=multilabel) for row in batch if row]
            y, x, x_rev, seq_lengths = zip(*data)
            if not max_len is None:
                x = pp.pad_sequences(x, maxlen=max_len, padding='post')
                x_rev = pp.pad_sequences(x_rev, maxlen=max_len, padding='post')
            else:
                x = pp.pad_sequences(x, padding='post')
                x_rev = pp.pad_sequences(x_rev, padding='post')
            yield np.array(y), x, x_rev, np.array(seq_lengths)
