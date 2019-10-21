import os
from math import ceil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from torch.utils.data.dataloader import DataLoader

from config import FEATURES_DATA_PATH, RESNET_MIN_DIM


# def get_shuffle_split(self, n_splits=2, test_size=0.5, train_size=0.5):
#     """
#     Return a generator to get a shuffle split of the data.
#     :param n_splits:
#     :param test_size:
#     :param train_size:
#     :return: x_train, y_train, x_test, y_test
#     """
#     print('info: starting shuffle-split training...')
#     kf = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size)
#     for train_index, test_index in kf.split(self.X):
#         yield self.X[train_index], self.Y[train_index], self.X[test_index], self.Y[test_index]


class DataManager(torch.utils.data.Dataset):
    def __init__(self, feature_name, data_type, feature_data_path, filenames=None, labels=None, label_encoder=None,
                 **kwargs):
        """

        :param data_type: [String] can be train, test or dev
        """
        self.feature_name = feature_name
        self.data_type = data_type
        self.feature_data_path = feature_data_path / self.feature_name
        self.cached_files = []
        # placeholders until data is loaded
        self.data_loader = None
        self.X, self.Y = None, None
        # batch settings (training)
        # load metadata
        labels_df = pd.read_csv(
            self.feature_data_path /
            'labels.{}.csv'.format(self.feature_name)
        )
        # parse metadata
        self.filenames = filenames if filenames is not None else labels_df['filename']
        self.labels = labels if labels is not None else labels_df['label']
        self.labels = np.asarray(labels)
        if len(self.labels.shape) == 1:
            # add one axis for actual label
            self.labels = self.labels.reshape(-1, 1)
        # fit label encoder with all annotations if available
        self.label_encoder = label_encoder if label_encoder else None
        self.label_encoder = self.label_encoder.fit(self.labels) if self.label_encoder else None

    def __len__(self):
        assert len(self.labels) == len(self.filenames)
        return len(self.labels)

    def __getitem__(self, idx):
        """

        :param idx: Integer to index one element
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.Y = np.asarray(self.labels)
        item_data = np.load(self.feature_data_path / self.filenames.iloc[idx])
        item_label = np.asarray(self.labels[idx])

        sample = {'data': item_data, 'label': item_label}

        if self.transform:
            sample = self.transform(sample)


        # sample = {'data': torch.Tensor(sample['data']), 'label': torch.Tensor(sample['label'])}
        return sample

    @classmethod
    def init_n_split(cls, feature_name, feature_data_path=FEATURES_DATA_PATH, shuffle=True, ratio=(0.5, 0.3, 0.2),
                     random_state=None,
                     **kwargs):
        """
        Split data into Train, Test and Dev set instantly.
        Instanciate 3 DataManger objects which are returned.

        :return: Train_DataManager, Test_Datamanager, Dev_DataManager
        """
        print('info: loading feature metadata from disk...')
        labels_df = pd.read_csv(
            feature_data_path /
            feature_name /
            'labels.{}.csv'.format(feature_name)
        )
        filenames = labels_df['filename']
        labels = labels_df['label']
        print('info: starting split...')
        assert ratio[0] + ratio[1] + ratio[2] == 1

        filenames_train, filenames_test, labels_train, labels_test = train_test_split(
            filenames, labels, test_size=ratio[1] + ratio[2], random_state=random_state, shuffle=shuffle
        )
        test_dev_pivot = round(ratio[1] / (ratio[1] + ratio[2]) * len(filenames_test))

        filenames_dev, labels_dev = filenames_test[test_dev_pivot:], labels_test[test_dev_pivot:]
        filenames_test, labels_test = filenames_test[:test_dev_pivot], labels_test[:test_dev_pivot]

        train_data_manager = cls(feature_name, 'train', feature_data_path=feature_data_path, filenames=filenames_train,
                                 labels=labels_train,
                                 **kwargs)
        test_data_manager = cls(feature_name, 'test', feature_data_path=feature_data_path, filenames=filenames_test,
                                labels=labels_test,
                                **kwargs)
        dev_data_manager = cls(feature_name, 'dev', feature_data_path=feature_data_path, filenames=filenames_dev,
                               labels=labels_dev,
                               **kwargs)

        train_data_manager.load_all(lazy=True, cache=True,
                                    ratio=ratio[0],
                                    random_state=random_state, )
        test_data_manager.load_all(lazy=True, cache=True,
                                   ratio=ratio[1],
                                   random_state=random_state, )
        dev_data_manager.load_all(lazy=True, cache=False)

        return train_data_manager, test_data_manager, dev_data_manager

    def get_cache_paths(self, *args):
        """
        Build cache name and paths, according to args
        :param args: List of interesting values to be encoded in the name.
        :return:
        """

        def _get_name(*args):
            return '_'.join(map(str, [*args]))

        x_path = self.feature_data_path / 'x_{}.npy'.format(_get_name(*args))
        y_path = self.feature_data_path / 'y_{}.npy'.format(_get_name(*args))
        self.cached_files.append(x_path)
        self.cached_files.append(y_path)
        return x_path, y_path

    def clean_cache(self):
        """
        Remove caches generated by this data manager
        :return:
        """
        c = 0
        while c < len(self.cached_files):
            file_path = self.cached_files[c]
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
            c += 1

    def transform(self, sample, **kwargs):
        """
        Transform a single sample to it's model compatible input form
        :param sample:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def load_all(self, lazy=False, **kwargs):
        """
        @deprecated use data loader instead
        Load all data to RAM.

        Warning: If lazy is False, calling this method may take several minutes to load.

        self.x: Numpy array like, most-probably float32 multidimentional sequential data.
        self.y: Numpy-array like, most-probably multi-cathegorical string labels.
        :return: If lazy is True, return the lazy reference to the function call
        """

        def _load_all(self, cache=True, **kwargs):
            """
            This is the real function to load all.
            :param self: Reference to object.
            :param cache: Flag to use cache.
            :param filenames: (List of strings) Manual input for .npy filenames of the data (X)
            :param labels: (List of strings) Manual input for labels of the data (Y)
            :param kwargs: Adtitional parameters also for formatting.
            :return:
            """
            # try to load from cache
            x_cache_path, y_cache_path = self.get_cache_paths(self.feature_name, type(self).__name__, self.data_type,
                                                              *kwargs.values())
            if cache:
                try:
                    self.X = np.load(x_cache_path)
                    self.Y = np.load(y_cache_path)
                    return
                except IOError:
                    pass
            print('info: loading data from disk...')
            print('warning: this operation takes some time. Go grab a tea...')

            # load hard data
            self.Y = np.asarray(self.labels)
            self.X = np.asarray([np.load(self.feature_data_path / filename) for filename in self.filenames])
            assert len(self.X) == len(self.Y)
            # if data is loaded, then format it and save cache
            if len(self.X) != 0:
                # apply formats
                self.format_all(**kwargs)
                # save cache
                np.save(x_cache_path, self.X) if cache else None
                np.save(y_cache_path, self.Y) if cache else None

        self.data_loader = lambda: _load_all(self, **kwargs)
        if not lazy:
            self.data_loader()

    def Y_to_one_shot(self):
        """
        @deprecated
        Y = ['foo', 'foo', 'bar']
        to
        Y = [['foo'], ['foo'], ['bar']]
        to
        Y = [[1,0], [1,0], [0,1]]
        :return: None
        """
        enc = OneHotEncoder()
        self.Y = enc.fit_transform(self.Y.reshape(-1, 1)).toarray()

    def Y_to_ordinal(self):
        """
        @deprecated
        Y = ['foo', 'foo', 'bar']
        to
        Y = [['foo'], ['foo'], ['bar']]
        to
        Y = [[0],[0],[1.0]]
        to
        Y = [0,0,1]
        :return: None
        """
        enc = OrdinalEncoder()
        self.Y = enc.fit_transform(self.Y.reshape(-1, 1))
        self.Y = np.array(self.Y.reshape(self.Y.shape[:-1]), dtype=np.int32)  # drop last axis and cast to int32

    def batch_iterator(self, max_step=None):
        """
        @deprecated
        Each iteration returns a batch of data with the following shape:
            (batch_size, *data.shape[1:])

        The data elements can be any python object.
        The parsing logic of this data is in get_feed_dict.


        :param max_step: limit number of data batches to be iterated, counted over all epochs
        :return: batch_data: custom objects with batch_data
            total_batch_num: total batch number
            epoch_idx: count of times all data has passed
            batch_idx: batch index
        """
        raise NotImplementedError()

    @staticmethod
    def get_feed_dict(model, batch_data, data_type='train'):
        """
        @deprecated
        Instance tf.variable values from batch_data, return the values in a
        TF compat Feed Dictionary.

        This method unify the parsing of the custom data to a standarized input for the NN.

        The returned feed_dict should include:
            @deprecated: self.token_seq: index of embedding: batch_size, max_length
            self.embedding_seq: sequence embeddings # batch_size, max_sequence_len, embedding_size
            self.output_labels integer from 0 to class_number: (batch_size)
            self.is_train True or False depending if it's training


        :param batch_data: A batch of data objects from batch_iterator method.
        :param data_type: String flag to tell if training or not
        :return: feed_dict with gathered values
        """
        raise NotImplementedError()

    @property
    def sample_num(self):
        """
        @deprecated
        Return the total number of loaded data samples.
        :return:
        """
        assert self.X.shape[0] == self.Y.shape[0]
        return self.X.shape[0]

    def format_all(self, **kwargs):
        """
        @Deprecated
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def get_dataloader(self, batch_size, shuffle=True):
        self.data_loader = None
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=2)

class ResnetDataManager(DataManager):

    def __init__(self, feature_name, data_type, feature_data_path, filenames=None, labels=None,
                 label_encoder=OneHotEncoder(),
                 **kwargs):
        super().__init__(feature_name, data_type, feature_data_path, filenames, labels, label_encoder, **kwargs)

    def transform(self, sample, **kwargs):
        """

        :param sample: dict with data and label keys.
        :return:
        """
        data, label = sample['data'], sample['label']
        if len(data.shape) == 2:
            # add channels dimension
            data = np.expand_dims(data, axis=-1)  # now has W, H, Channels
        assert len(data.shape) == 3  # W, H, Channels
        # padding if necessary
        dim_1_offset = RESNET_MIN_DIM - data.shape[0]
        dim_2_offset = RESNET_MIN_DIM - data.shape[1]
        data = np.pad(data, (
            (0, max(0, dim_1_offset)),
            (0, max(0, dim_2_offset)),
            (0, 0)),
                        'reflect')
        # encode single label with encoder
        label = self.label_encoder.transform(label.reshape(1, -1)).toarray() if self.label_encoder else label
        return {'data': data, 'label': label}

    def format_all(self, **kwargs):
        """
        @deprecated
        Format loaded data according to model input layout.
        self.X shape is #_data, *dims
        self.y shape is #_data, 1 with label as string
        :return:
        """
        # assert this method is called after load_all with these:
        assert self.X is not None
        assert self.Y is not None

        print('info: formatting data...')
        if len(self.X.shape) == 3:
            # as ain image has channels, this visual classificator expects at least 1 channel,
            # this is represented as a the fourth dim: #_data, W, H, Channels
            self.X = np.expand_dims(self.X, axis=-1)
        self.Y_to_one_shot()

        # padding if neccesary
        assert len(self.X.shape) == 4
        dim_1_offset = RESNET_MIN_DIM - self.X.shape[1]
        dim_2_offset = RESNET_MIN_DIM - self.X.shape[2]
        self.X = np.pad(self.X, (
            (0, 0),
            (0, max(0, dim_1_offset)),
            (0, max(0, dim_2_offset)),
            (0, 0)),
                        'reflect')

        self.X = self.X.reshape((self.X.shape[0], -1)) if kwargs.get('flatten', None) else self.X

        print('info: done formatting...')


class TorchVisionDataManager(DataManager):
    def __init__(self, feature_name, data_type, feature_data_path, **kwargs):
        super().__init__(feature_name, data_type, feature_data_path, **kwargs)

    def format_all(self, **kwargs):
        """
        Format loaded data according to model input layout.
        :return:
        """
        # assert this method is called after load_all with these:
        assert self.X is not None
        assert self.Y is not None

        print('info: formatting data...')
        if len(self.X.shape) == 3:
            # as ain image has channels, this visual classificator expects at least 1 channel,
            # this is represented as a the fourth dim: #_data, W, H, Channels
            self.X = np.expand_dims(self.X, axis=-1)
        self.Y_to_ordinal()
        self.Y = self.Y.astype(np.int64)

        # padding if neccesary
        assert len(self.X.shape) == 4
        dim_1_offset = RESNET_MIN_DIM - self.X.shape[1]
        dim_2_offset = RESNET_MIN_DIM - self.X.shape[2]
        self.X = np.pad(self.X, (
            (0, 0),
            (0, max(0, dim_1_offset)),
            (0, max(0, dim_2_offset)),
            (0, 0)),
                        'reflect')
        self.X = np.moveaxis(self.X, -1, 1)  # move channels to second position to match pythorch standard
        self.X = self.X.reshape((self.X.shape[0], -1)) if kwargs.get('flatten', None) else self.X

        print('info: done formatting...')


class SSTDataManager(DataManager):
    pass


class ADiSANDataManager(DataManager):

    def __init__(self, feature_name, data_type, feature_data_path, **kwargs):
        super().__init__(feature_name, data_type, feature_data_path, **kwargs)

    def format_all(self, **kwargs):
        """
        Format loaded data according to model input layout.

        self.X: (#_data, #_sequence, #_feature) [dtype = float32]   Feature vectors of the Data
        self.Y: (#_data, )  [dtype = int32]   Id of the class
        :return:
        """
        # assert this method is called after load_all with these:
        assert self.X is not None
        assert self.Y is not None

        print('info: formatting data...')
        assert len(self.X.shape) == 3  # assert #_data, #_sequence, #_feature
        self.Y_to_ordinal()
        print('info: done formatting...')

    def batch_iterator(self, max_step=None):
        """
        Each iteration returns a batch of data with the following shape:
            (batch_size, *data.shape[1:])

        The data elements can be any python object.
        The parsing logic of this data is in get_feed_dict.


        :param max_step: limit number of data batches to be iterated, counted over all epochs. i.e. length of the iterator
        :return: batch_data: custom objects with batch_data
            total_batch_num: total batch number
            epoch_idx: count of times all data has passed
            batch_idx: batch index
        """
        # early stop by max_step
        stop = False
        step_counter = 0

        # helper vars
        n_data = self.X.shape[0]
        total_batch_count = ceil(n_data / self.batch_size)

        # batch loops
        for epoch_idx in range(self.epochs):
            if stop:
                # early stop by max_step
                break
            for batch_idx in range(total_batch_count):
                step_counter += 1
                if max_step and step_counter > max_step:
                    # early stop by max_step
                    stop = True
                    break
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                if end_idx <= n_data:
                    # the batch fits completely inside the data
                    x = self.X[start_idx:end_idx]
                    y = self.Y[start_idx:end_idx]

                else:
                    # Circular Padding:
                    # the batch need padding because
                    # there is no enough data in X & Y to fit batch_size perfectly
                    padding_size = end_idx - n_data
                    x = np.concatenate((self.X[start_idx:end_idx], self.X[:padding_size]))
                    y = np.concatenate((self.Y[start_idx:end_idx], self.Y[:padding_size]))

                batch_data = {
                    'x': x,
                    'y': y
                }
                yield batch_data, total_batch_count, epoch_idx, batch_idx

    @staticmethod
    def get_feed_dict(model, batch_data, data_type='train'):
        """
        Instance tf.variable values from batch_data, return the values in a
        TF compat Feed Dictionary.

        This method unify the parsing of the custom data to a standarized input for the NN.

        The returned feed_dict should include:
            @deprecated: self.token_seq: index of embedding: batch_size, max_length
            self.embedding_seq: sequence embeddings # batch_size, max_sequence_len, embedding_size
            self.output_labels integer from 0 to class_number: (batch_size)
            self.is_train True or False depending if it's training

        :param: model: A model instance.
        :param batch_data: A batch of data objects from batch_iterator method.
        :param data_type: String flag to tell if training or not
        :return: feed_dict with gathered values
        """
        batch_embedding_sequence = batch_data['x']  # (batch_size, seq_len, emb_dim)
        batch_output_labels = batch_data['y']  # (batch_size, )
        # in this case, we can suppose all the sequence are the same length
        # so they doesnt need special masking
        # ex. WindowedMelSpectralCoefficientsFeatureExtractor use windows of 1 second
        # resullting in a fixed shape of (128, 16)
        batch_access_mask = np.full(
            (batch_embedding_sequence.shape[0:-1]),
            True
        )  # (batch_size, seq_len)

        feed_dict = {model.batch_embedding_sequence: batch_embedding_sequence,
                     model.batch_output_labels: batch_output_labels,
                     model.batch_access_mask: batch_access_mask,
                     model.is_train: True if data_type == 'train' else False}
        return feed_dict
