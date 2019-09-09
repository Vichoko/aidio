import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from config import FEATURES_DATA_PATH


class DataManager:
    def __init__(self, feature_name, data_type, feature_data_path=FEATURES_DATA_PATH, **kwargs):
        """

        :param data_type: [String] can be train, test or dev
        """
        self.feature_name = feature_name
        self.data_type = data_type
        self.feature_data_path = feature_data_path / self.feature_name
        self.data_loader = None
        self.X, self.Y = None, None

    @property
    def sample_num(self):
        """
        Return the total number of loaded data samples.
        :return:
        """
        assert self.X.shape[0] == self.Y.shape[0]
        return self.X.shape[0]

    @classmethod
    def init_n_split(cls, feature_name, feature_data_path=FEATURES_DATA_PATH, shuffle=True, ratios=(0.5, 0.3, 0.2),
                     **kwargs):
        """
        Split data into Train, Test and Dev set instantly.
        Instanciate 3 DataManger objects which are returned.

        :return: Train_DataManager, Test_Datamanager, Dev_DataManager
        """
        random_state = 42
        print('info: loading feature metadata from disk...')
        labels_df = pd.read_csv(
            feature_data_path /
            'labels.{}.csv'.format(feature_name)
        )
        filenames = labels_df['filename']
        labels = labels_df['label']
        print('info: starting split...')
        assert ratios[0] + ratios[1] + ratios[2] == 1

        filenames_train, labels_train, filenames_test, labels_test = train_test_split(
            filenames, labels, test_size=ratios[1] + ratios[2], random_state=random_state, shuffle=shuffle
        )
        filename_pivot, labels_pivot = int(len(filenames_test) * ratios[1]), \
                                       int(len(labels_test) * ratios[1])

        filenames_dev, labels_dev = filenames_test[filename_pivot:], labels_test[labels_pivot:]
        filenames_test, labels_test = filenames_test[:filename_pivot], labels_test[:labels_pivot]

        train_data_manager = cls(feature_name, 'train', feature_data_path, **kwargs)
        test_data_manager = cls(feature_name, 'test', feature_data_path, **kwargs)
        dev_data_manager = cls(feature_name, 'dev', feature_data_path, **kwargs)

        train_data_manager.load_all(lazy=False, cache=False, filenames=filenames_train, labels=labels_train)
        test_data_manager.load_all(lazy=True, cache=False, filenames=filenames_test, labels=labels_test)
        dev_data_manager.load_all(lazy=True, cache=False, filenames=filenames_dev, labels=labels_dev)

        return train_data_manager, test_data_manager, dev_data_manager

    def load_all(self, lazy=False, **kwargs):
        """
        Warning: Can take several minutes to load.
        Load all data to RAM.

        self.x: Numpy array like, most-probably float32 multidimentional sequential data.
        self.y: Numpy-array like, most-probably multi-cathegorical string labels.
        :return: If lazy is True, return the lazy reference to the function call
        """

        def _load_all(self, cache=True, filenames=None, labels=None, **kwargs):
            x_cache_file_name = 'x_{}_{}.npy'.format(self.feature_name, self.data_type)
            y_cache_file_name = 'y_{}_{}.npy'.format(self.feature_name, self.data_type)
            if cache:
                try:
                    self.X = np.load(self.feature_data_path / x_cache_file_name)
                    self.Y = np.load(self.feature_data_path / y_cache_file_name)
                    return
                except IOError:
                    pass
            print('info: loading data from disk...')
            print('warning: this operation takes some time. Go grab a tea...')
            labels_df = pd.read_csv(
                self.feature_data_path /
                'labels.{}.csv'.format(self.feature_name)
            )
            filenames = filenames or labels_df['filename']
            labels = labels or labels_df['label']
            self.Y = np.asarray(labels)
            self.X = np.asarray([np.load(self.feature_data_path / filename) for filename in filenames])
            self.format_all(**kwargs)
            np.save(self.feature_data_path / x_cache_file_name, self.X) if cache else None
            np.save(self.feature_data_path / y_cache_file_name, self.Y) if cache else None

        self.data_loader = lambda: _load_all(self, **kwargs)
        if not lazy:
            self.data_loader()

    def Y_to_one_shot(self):
        enc = OneHotEncoder()
        self.Y = enc.fit_transform(self.Y.reshape(-1, 1)).toarray()

    def format_all(self, **kwargs):
        raise NotImplementedError()

    def batch_iterator(self, max_step=None):
        """
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

    def get_feed_dict(self, batch_data, data_type='train'):
        """
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


class ResnetDataManager(DataManager):

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
        self.Y_to_one_shot()
        self.X = self.X.reshape((self.X.shape[0], -1)) if kwargs.get('flatten', None) else self.X
        print('info: done formatting...')


class SSTDataManager(DataManager):
    pass
