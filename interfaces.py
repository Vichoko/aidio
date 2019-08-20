import os
import concurrent.futures

import librosa

from config import CPU_WORKERS, FEATURES_DATA_PATH, RAW_DATA_PATH, makedirs, SR, AVAIL_MEDIA_TYPES

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
    feature_name = 'UnnamedFeature'
    dependency_feature_name = ''

    def __init__(self, x, y, out_path=FEATURES_DATA_PATH, raw_path=RAW_DATA_PATH):
        self.x = x
        self.y = y
        self.out_path = out_path / self.feature_name
        makedirs(self.out_path)
        self.raw_path = raw_path
        self.new_labels = []
        self.trigger_dependency_warnings_if_needed()

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

    def trigger_dependency_warnings_if_needed(self):
        """
        Sanity Check for expected input for a particular Feature Extractor,
        defined by its dependency_feature_name, compared to the path of the raw_path property.

        If dependency_feature_name isn't set, it checks that the input path and filenames are
        audifiles.

        Also compares the stored filenames in self.x for the dependency_feature_name File Format.
        :return: (input_warning, filename_warning) [tuple of booleans] True if corresponding warning
                        was triggered in this method. A warning is printed when this flag is True.
        """
        input_path_warning_flag = False
        filename_format_warning_flag = False
        if self.dependency_feature_name:
            # """
            # If this parameter is given, the input is a feature in the Feature folder.
            # """
            if self.raw_path != (FEATURES_DATA_PATH / self.dependency_feature_name):
                print("""warning: 
                {} Feature source folder is commonly FEATURES_DATA_PATH/{} config
                because it need {} feature as source, receeived {} instead.""".format(self.feature_name,
                                                                                      self.dependency_feature_name,
                                                                                      self.dependency_feature_name,
                                                                                      self.raw_path))
            input_path_warning_flag = True
            if '.{}.npy'.format(self.dependency_feature_name) not in self.x[0]:
                print("""warning:
                {} Feature source filenames are commonly formatted like <name>.{}.npy, received {} instead
                """.format(self.feature_name, self.dependency_feature_name, self.x[0]))
                filename_format_warning_flag = True
        else:
            # """
            # If this parameter isn't set, the raw_path should be the RAW_DATA_PATH,
            # also the files should end in an accepted format.
            # """
            if self.raw_path != RAW_DATA_PATH:
                print('warning: this FeatureExtractor has a modified self.raw_path ({}), '
                      'but self.dependency_feature_name wasn\'t set. If this path doesn\t contain any audio files, '
                      'this extractor will probably fail.'
                      'Prefer to set RAW_DATA_PATH for using audio files.'.format(self.raw_path))
                input_path_warning_flag = True
            if self.x[0].split('.')[-1] not in AVAIL_MEDIA_TYPES:
                print('warning: No self.dependency_feature_name was set, and '
                      'the parsed filenames (self.x) has an unsupported media type ({}) '.format(
                    self.x[0].split('.')[-1]))
                filename_format_warning_flag = True
        return input_path_warning_flag, filename_format_warning_flag

    def clean_references(self):
        """
        Remove elements from x and y that doesnt't have a source (raw) file
        :param x: from self.x; a list of filenames (str)
        :param y: from self.y; a list of labels (str)
        :param raw_path: pathlib.Path object of where source files are located
        :return: cleansed x and y
        """
        new_x = []
        new_y = []
        for i, x_i in enumerate(self.x):
            if os.path.exists(self.raw_path / x_i):
                y_i = self.y[i]
                new_x.append(x_i)
                new_y.append(y_i)
        self.x = new_x
        self.y = new_y

    @staticmethod
    def process_element(feature_name, new_labels, out_path, raw_path, **kwargs):
        def __process_element(data):
            """
            :param x: filename (str)
            :param y: label (str)
            :return:
            """
            print('prosessing {}'.format(data))
            x = data[0]
            y = data[1]
            # foo
            product = 'foo'

            # this is kind-of standard
            name = '.'.join(x.split('.')[:-1])
            filename = '{}.{}.npy'.format(name, feature_name)
            np.save(out_path / filename, product)  # replace x with product
            new_labels.append([filename, y])
            print('info: {} transformed and saved!'.format(filename))
            raise NotImplemented

        # stub
        return __process_element

    @staticmethod
    def proccess_elements(feature_name, new_labels, out_path, raw_path, fun=None, **kwargs):
        def __process_elements(data):
            for data_element in data:
                fun(data_element)

        return __process_elements

    def parallel_transform(self, parallel=True, **kwargs):
        self.clean_references()
        data = np.asarray([self.x, self.y]).swapaxes(0, 1)

        fun = self.process_element(feature_name=self.feature_name, new_labels=self.new_labels, out_path=self.out_path,
                                   raw_path=self.raw_path, **kwargs)
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
                executor.map(fun, data)
        else:
            fun = self.proccess_elements(feature_name=self.feature_name, new_labels=self.new_labels,
                                         out_path=self.out_path,
                                         raw_path=self.raw_path, fun=fun, **kwargs)
            fun(data)

        df = pd.DataFrame(np.asarray(self.new_labels))
        df.columns = ['filename', 'label']
        df.to_csv(self.out_path / 'labels.{}.csv'.format(self.feature_name), index=False)

    @staticmethod
    def get_file_name(x, feature_name, ext='npy'):
        """
        Feature File System logic is here
        :param ext:
        :param x:
        :param feature_name:
        :return:
        """
        # this is kind-of standard
        name = '.'.join(x.split('.')[:-1])
        filename = '{}.{}.{}'.format(name, feature_name, ext)
        return filename

    @staticmethod
    def save_feature(ndarray, feature_name, out_path, x, y, new_labels, filename=None):
        """
        Save any numpy object in Feature File System.
        :param ndarray:
        :param feature_name:
        :param out_path:
        :param x:
        :param filename:
        :return:
        """
        # this is kind-of standard
        filename = filename or FeatureExtractor.get_file_name(x, feature_name)
        np.save(out_path / filename, ndarray)
        new_labels.append([filename, y])
        print('info: {} transformed and saved!'.format(filename))
        return filename

    @staticmethod
    def save_audio(ndarray, feature_name, out_path, x, y, new_labels, filename=None):
        """
        Save any numpy object in Feature File System.
        :param ndarray:
        :param feature_name:
        :param out_path:
        :param x:
        :param filename:
        :return:
        """
        # this is kind-of standard
        filename = filename or FeatureExtractor.get_file_name(x, feature_name, 'wav')
        librosa.output.write_wav(out_path / filename, ndarray, sr=SR, norm=True)
        new_labels.append([filename, y])
        print('info: {} transformed and saved!'.format(filename))
        return filename
