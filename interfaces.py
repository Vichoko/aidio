import threading
import concurrent.futures
from config import CPU_WORKERS, FEATURES_DATA_PATH, RAW_DATA_PATH

from functools import partial

import pandas as pd
import numpy as np


class ClassificationModel:
    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        return

    def data_loader(self, audio_data, label_data=None):
        raise NotImplemented

    def train(self, audio_data, label_data, options):
        """
        Train model specified by options with given data.

        :param audio_data: iterable reference
        :param label_data: iterable reference
        :param options: dict-like; model dependant (cnn, aidsan, etc)
        :return:
        """
        self.x, self.y = self.data_loader(audio_data, label_data)
        return

    def predict(self, audio_data, options):
        """
        Predict with given data, and options.

        :param audio_data: reference
        :param options:
        :return:
        """
        self.x = self.data_loader(audio_data)
        return


class AudioButcher:
    def data_loader(self, data):
        """
        load iterable of references to actual manegable objects
        :param data:
        :return:
        """
        raise NotImplemented

    def transform(self, data, options):
        """

        :param data: iterable of references
        :param options:
        :return:
        """
        raise NotImplemented


class FeatureExtractor:
    def __init__(self, x, y, out_path=FEATURES_DATA_PATH, raw_path=RAW_DATA_PATH):
        self.x = x
        self.y = y
        self.out_path = out_path
        self.raw_path = raw_path
        self.new_labels = []

    @classmethod
    def from_label_file(cls, label_file_path, out_path=FEATURES_DATA_PATH, raw_path=RAW_DATA_PATH):
        """
        Initiate extractor fron label file (csv) that contains
        filename and label columns.

        :param label_file_path:
        :return:
        """
        df = pd.read_csv(label_file_path)
        filenames = df['filename']
        labels = df['label']
        return cls(filenames, labels, out_path=out_path, raw_path=raw_path)

    @staticmethod
    def process_element(**kwargs):
        raise NotImplemented

    def parallel_transform(self, feature_name, new_labels, parallel=True, **kwargs):
        data = np.asarray([self.x, self.y]).swapaxes(0, 1)

        fun = self.process_element(feature_name=feature_name, new_labels=new_labels, **kwargs)
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
                executor.map(fun, data)
        else:
            for data_element in data:
                fun(data_element)

        df = pd.DataFrame(np.asarray(new_labels))
        df.columns = ['filename', 'label']

        df.to_csv(self.out_path / 'new_labels.{}.npy'.format(feature_name), index=False)
