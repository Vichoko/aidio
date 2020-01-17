import os
from math import ceil

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from config import FEATURES_DATA_PATH, RESNET_MIN_DIM, ADISAN_BATCH_SIZE, ADISAN_EPOCHS, WAVEFORM_MAX_SEQUENCE_LENGTH, \
    WAVEFORM_NUM_CHANNELS, WAVEFORM_SAMPLE_RATE
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
from features import MelSpectralCoefficientsFeatureExtractor


# import torch


class DataManager:
    def __init__(self, feature_name, data_type, batch_size, epochs, feature_data_path, **kwargs):
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
        self.batch_size = batch_size
        self.epochs = epochs

    @property
    def sample_num(self):
        """
        Return the total number of loaded data samples.
        :return:
        """
        assert self.X.shape[0] == self.Y.shape[0]
        return self.X.shape[0]

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

        train_data_manager = cls(feature_name, 'train', feature_data_path=feature_data_path, batch_size=1, epochs=1,
                                 **kwargs)
        test_data_manager = cls(feature_name, 'test', feature_data_path=feature_data_path, batch_size=1, epochs=1,
                                **kwargs)
        dev_data_manager = cls(feature_name, 'dev', feature_data_path=feature_data_path, batch_size=1, epochs=1,
                               **kwargs)

        train_data_manager.load_all(lazy=False, cache=True, filenames=filenames_train,
                                    labels=labels_train,
                                    ratio=ratio[0],
                                    random_state=random_state, )
        test_data_manager.load_all(lazy=True, cache=True, filenames=filenames_test,
                                   labels=labels_test,
                                   ratio=ratio[1],
                                   random_state=random_state, )
        dev_data_manager.load_all(lazy=True, cache=False, filenames=filenames_dev, labels=labels_dev)

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

    def load_all(self, lazy=False, **kwargs):
        """
        Load all data to RAM.

        Warning: If lazy is False, calling this method may take several minutes to load.

        self.x: Numpy array like, most-probably float32 multidimentional sequential data.
        self.y: Numpy-array like, most-probably multi-cathegorical string labels.
        :return: If lazy is True, return the lazy reference to the function call
        """

        def _load_all(self, cache=True, filenames=None, labels=None, **kwargs):
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
            # load metadata
            labels_df = pd.read_csv(
                self.feature_data_path /
                'labels.{}.csv'.format(self.feature_name)
            )
            # parse metadata
            filenames = filenames if filenames is not None else labels_df['filename']
            labels = labels if labels is not None else labels_df['label']
            # load hard data
            self.Y = np.asarray(labels)
            self.X = np.asarray([np.load(self.feature_data_path / filename) for filename in filenames])
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


        :param batch_data: A batch of data objects from batch_iterator method.
        :param data_type: String flag to tell if training or not
        :return: feed_dict with gathered values
        """
        raise NotImplementedError()


class ResnetDataManager(DataManager):
    def __init__(self, feature_name, data_type, batch_size=None, epochs=None, feature_data_path=FEATURES_DATA_PATH,
                 **kwargs):
        super().__init__(feature_name, data_type, batch_size, epochs, feature_data_path, **kwargs)

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
    def __init__(self, feature_name, data_type, batch_size=None, epochs=None, feature_data_path=FEATURES_DATA_PATH,
                 **kwargs):
        super().__init__(feature_name, data_type, batch_size, epochs, feature_data_path, **kwargs)

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

    def __init__(self, feature_name, data_type, batch_size=ADISAN_BATCH_SIZE, epochs=ADISAN_EPOCHS,
                 feature_data_path=FEATURES_DATA_PATH, **kwargs):
        super().__init__(feature_name, data_type, batch_size, epochs, feature_data_path, **kwargs)
        self.batch_size = batch_size

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


class ExperimentDataset(Dataset):
    def __init__(self, filenames, labels, data_path, label_encoder) -> None:
        """
        Create a dataset specification with the giver parameters
        :param filenames:
        :param labels: Array like with the raw labels as loaded from metadata file
        :param data_path: Path to where the feature folders are stored.
        :param transform:
        :param label_encoder: Pytorch encoder if needed for the self.encode_labels method.
        """
        self.data_path = data_path
        self.filenames = np.asarray(filenames)
        self.labels = self.encode_labels(np.asarray(labels), label_encoder)
        # self.transform = transforms.Compose(
        #     []
        # )
        assert len(self.filenames) == len(self.labels)
        super().__init__()

    def __len__(self) -> int:
        return len(self.filenames)

    @classmethod
    def init_sets(cls, data_path, label_filename,
                  ratio=(0.5, 0.3, 0.2),
                  shuffle=True,
                  random_state=None, ):
        """
        Initiate 3 Datasets: Train, Validation and Test, splitted by the given ratios.
        :param feature_name:
        :param feature_path:
        :param shuffle:
        :param ratio:
        :param random_state:
        :return:
        """

        # metadata_df = pd.read_csv(
        #     features_path /
        #     feature_name /
        #     'labels.{}.csv'.format(feature_name)
        # )

        metadata_df = pd.read_csv(data_path / label_filename)
        filenames = metadata_df['filename']
        labels = metadata_df['label']
        print('info: starting split...')
        assert ratio[0] + ratio[1] + ratio[2] == 1

        # check the nmber of classes and fit an ordinal ecoder
        labels = np.asarray(labels)
        number_of_classes = len(set(labels))
        label_encoder = OrdinalEncoder().fit(labels.reshape(-1, 1))

        # split metadata in 3 sets: train, test and dev
        filenames_train, filenames_test, labels_train, labels_test = train_test_split(
            filenames, labels, test_size=ratio[1] + ratio[2], random_state=random_state, shuffle=shuffle
        )
        test_dev_pivot = round(ratio[1] / (ratio[1] + ratio[2]) * len(filenames_test))
        filenames_dev, labels_dev = filenames_test[test_dev_pivot:], labels_test[test_dev_pivot:]
        filenames_test, labels_test = filenames_test[:test_dev_pivot], labels_test[:test_dev_pivot]

        # instance 3 datasets
        train_dataset = cls(
            filenames_train,
            labels_train,
            data_path,
            label_encoder
        )
        test_dataset = cls(
            filenames_test,
            labels_test,
            data_path,
            label_encoder
        )
        dev_dataset = cls(
            filenames_dev,
            labels_dev,
            data_path,
            label_encoder
        )
        return train_dataset, test_dataset, dev_dataset, number_of_classes

    @staticmethod
    def encode_labels(labels, label_encoder):
        """
        Function that transform the raw labels to another array like labels.
        :param labels: Array like with the raw labels as loaded from metadata file.
        :param label_encoder:Pytorch encoder if needed.
        :return: Array-like with new labels as needed.
        """
        print('warning: using default encode_labels method. Forwarding raw labels...')
        return labels


class WaveformDataset(ExperimentDataset):
    """
    Load the input data as a wave net with a specified sample rate.
    """

    def __init__(self, filenames, labels, data_path, label_encoder) -> None:
        super().__init__(
            filenames,
            labels,
            data_path,
            label_encoder
        )
        self.transform = transforms.Compose(
            [self.RandomCrop(WAVEFORM_MAX_SEQUENCE_LENGTH),
             self.ToTensor()]
        )

    def __getitem__(self, index: int):
        label = self.labels[index]
        wav, sr = librosa.load(
            str(self.data_path / self.filenames[index]),
            sr=WAVEFORM_SAMPLE_RATE,
            mono=True if WAVEFORM_NUM_CHANNELS == 1 else WAVEFORM_NUM_CHANNELS
        )

        # wav shape is (n_samples) or (n_channels, n_samples)
        # torch 1d image is n_channels, n_samples
        if len(wav.shape) == 1:
            wav = wav.reshape(1, -1)
        elif len(wav.shape) == 2:
            if wav.shape[0] > 2:
                print("warning: wav channel size is uncommon {}; expected is 2 (stereo) or 1 (mono) ".format(
                    wav.shape[0]))
        else:
            print('error: wav shape is {}, expected N_channels x N_samples'.format(wav.shape))
        # unify wav shape to (n_channels, n_samples)

        sample = {'x': wav, 'y': label}
        sample = self.transform(sample)
        return sample

    @staticmethod
    def encode_labels(labels, label_encoder=None):
        """
        Y = ['foo', 'foo', 'bar']
        to
        Y = [['foo'], ['foo'], ['bar']]
        to
        Y = [[0],[0],[1.0]]
        to
        Y = [0,0,1]
        :return: None
        """
        labels = np.asarray(labels)
        if label_encoder is None:
            """
            OrdinalEncoder is used for CE Loss criteria 
            """
            label_encoder = OrdinalEncoder()
            label_encoder.fit(labels.reshape(-1, 1))
        labels = label_encoder.transform(labels.reshape(-1, 1))
        labels = np.array(labels.reshape(labels.shape[:-1]),
                          dtype=np.int64)  # drop last axis and cast to int64 aka long
        return labels

    class ToTensor:
        def __call__(self, sample):
            wav, label = sample['x'], sample['y']
            return {'x': torch.from_numpy(wav), 'y': torch.from_numpy(np.asarray(label))}

    class RandomCrop:
        """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

        def __init__(self, output_size):
            assert isinstance(output_size, int)
            if isinstance(output_size, int):
                self.output_size = output_size

        def __call__(self, sample):
            wav, label = sample['x'], sample['y']
            # wav shape is n_channels, n_samples
            l = wav.shape[1]
            new_l = self.output_size
            max_pivot_exclusive = l - new_l
            print("warning: RandomCrop processing a WAV with 0-length") if l <= 0 else None
            # if wav length is greater than new_length, random chose a pivot and pick random new_length sub-sequence.
            # if wav length is less than new_length, then just grab all the wav from the beggining.
            pivot = np.random.randint(0, max_pivot_exclusive) if max_pivot_exclusive > 0 else 0
            wav = wav[:, pivot: pivot + new_l]
            l = wav.shape[1]
            # padding if neccesary
            if l < new_l:
                print('debug: padding {} of len {}'.format(label, wav.shape[1]))
                # this ad-hoc padding just repeat the beggining of the wav until the sequnece is long enough for the model
                wav = torch.cat((wav, wav[:, new_l - l]), 2)
            return {'x': wav, 'y': label}


class CepstrumDataset(ExperimentDataset):
    """
    Load the input data as an MFCC with specified number of cepstral coefficients.
    """

    input_shape = WAVEFORM_MAX_SEQUENCE_LENGTH

    def __init__(self, filenames, labels, data_path, label_encoder) -> None:
        super().__init__(
            filenames,
            labels,
            data_path,
            label_encoder
        )
        self.transform = transforms.Compose([
            # [torchvision.transforms.RandomCrop(cls.input_shape),
            #  ]  # with numpy
            self.ToTensor()]  # with torch
        )

    def __getitem__(self, index: int):
        label = self.labels[index]
        data = np.load(
            str(self.data_path / self.filenames[index]),
            allow_pickle=True
        )

        sample = {'x': data, 'y': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def encode_labels(labels, label_encoder=None):
        """
        Y = ['foo', 'foo', 'bar']
        to
        Y = [['foo'], ['foo'], ['bar']]
        to
        Y = [[0],[0],[1.0]]
        to
        Y = [0,0,1]
        :return: None
        """
        labels = np.asarray(labels)
        if label_encoder is None:
            """
            OrdinalEncoder is used for CE Loss criteria 
            """
            label_encoder = OrdinalEncoder()
            label_encoder.fit(labels.reshape(-1, 1))
        labels = label_encoder.transform(labels.reshape(-1, 1))
        labels = np.array(labels.reshape(labels.shape[:-1]),
                          dtype=np.int64)  # drop last axis and cast to int64 aka long
        return labels

    class ToTensor:
        def __call__(self, sample):
            feature_tensor, label = sample['x'], sample['y']
            feature_tensor = feature_tensor.reshape(1, feature_tensor.shape[0], -1)
            return {'x': torch.from_numpy(feature_tensor),
                    'y': torch.from_numpy(np.asarray(label)).type('torch.LongTensor')}


class ClassSampler(Sampler):
    def __init__(self, number_of_classes, labels, batch_size=None):
        super().__init__(None)
        self.number_of_classes = number_of_classes
        self.labels = np.asarray(labels)
        self.batch_size = batch_size

        if len(self.labels.shape) > 1:
            raise RuntimeError('labels should be a flattened array. Detected {}.'.format(len(self.labels.shape)))

        if number_of_classes < 1:
            raise RuntimeError('number of classes is < 1, this doesnt make sense')

    def __len__(self) -> int:
        """
        One batch per class.
        :return:
        """
        return self.number_of_classes

    def __iter__(self):
        """
        Iterate over possible classes, yielding the indices of the samples of that class.
        :yield: a list of indexes
        """
        for label in range(self.number_of_classes):
            relevant_indexes = (self.labels != label).nonzero()[0]

            if self.batch_size is None:
                yield relevant_indexes.tolist()
            else:
                # random sample without replacement
                yield np.random.choice(relevant_indexes, self.batch_size, replace=False)

    @staticmethod
    def collate_fn(batch):
        """
        It receives a batch of track mfcc coeffcieints.
        It supposes  all the received frames are relevant (so make preprocesses before)
        All the song frames are join in a single dimension as they belong to the same class.

        The batching logic is by classes so every track is from the same class.

        :param batch: samples of the same class
        :return: tuple of data Tensor, label Tensor
        """
        label = batch[0]['y']  # as each batch is the same class
        out_mfcc = None
        for batch_element in batch:
            batch_data = batch_element['x'].type(torch.FloatTensor)
            if out_mfcc is None:
                out_mfcc = batch_data
            else:
                out_mfcc = torch.cat((out_mfcc, batch_data), 2)
        out_mfcc = out_mfcc.squeeze()  # the first dimension (track) is dropped as all frames are concatenated (1, 128, n_frames)
        # final size is (128, n_total_frames)
        out_mfcc = out_mfcc.permute(1, 0)  # to (n_frames, n_features) standard format
        return {'x': out_mfcc, 'y': label}


def mfcc_test():
    train_dataset, test_dataset, eval_dataset, number_of_classes = CepstrumDataset.init_sets(
        MelSpectralCoefficientsFeatureExtractor.feature_name,
        FEATURES_DATA_PATH,
        ratio=(0.5, 0.25, 0.25)
    )

    train_dataloader = DataLoader(train_dataset, num_workers=4,
                                  batch_sampler=ClassSampler(number_of_classes, train_dataset.labels),
                                  collate_fn=ClassSampler.collate_fn)
    test_dataloader = DataLoader(test_dataset, num_workers=0,
                                 batch_sampler=ClassSampler(number_of_classes, test_dataset.labels))
    val_dataloader = DataLoader(eval_dataset, num_workers=0,
                                batch_sampler=ClassSampler(number_of_classes, eval_dataset.labels))

    for b in train_dataloader:
        print(b)
    print('end')


if __name__ == '__main__':
    mfcc_test()
