import argparse
import concurrent.futures
import os
import pathlib
from math import ceil

import librosa
import numpy as np
import pandas as pd
from librosa import frames_to_time

from config import SR, RAW_DATA_PATH, FEATURES_DATA_PATH, HOP_LENGTH, N_FFT, N_MELS, FMIN, FMAX, POWER, \
    VOICE_DETECTION_MODEL_NAME, N_FFT_HPSS_1, N_HOP_HPSS_1, N_FFT_HPSS_2, N_HOP_HPSS_2, SR_HPSS, \
    N_MELS_HPSS, MODELS_DATA_PATH, RNN_INPUT_SIZE_VOICE_ACTIVATION, TOP_DB_WINDOWED_MFCC, \
    MIN_INTERVAL_LEN_WINDOWED_MFCC, WINDOW_LEN_WINDOWED_MFCC, WINDOW_HOP_WINDOWED_MFCC, makedirs, AVAIL_MEDIA_TYPES, \
    CPU_WORKERS
from util.leglaive.audio import ono_hpss, log_melgram


# MFSC
class FeatureExtractor:
    feature_name = 'UnnamedFeature'
    dependency_feature_name = ''

    def __init__(self, x, y, out_path=FEATURES_DATA_PATH, source_path=RAW_DATA_PATH, feature_path=FEATURES_DATA_PATH,
                 raw_path=RAW_DATA_PATH):
        self.x = x
        self.y = y
        self.out_path = out_path / self.feature_name
        self.feature_path = feature_path
        self.raw_path = raw_path
        makedirs(self.out_path)
        self.source_path = source_path
        self.new_labels = []
        self.trigger_dependency_warnings_if_needed()
        self.trigger_dependency_extraction_if_needed()
        print('info: extractor initialized with following data {}'.format(self.x))

    def trigger_dependency_extraction_if_needed(self):
        if self.dependency_feature_name:
            dependency_extractor = AVAILABLE_FEATURES[self.dependency_feature_name]
            try:
                df = pd.read_csv(
                    self.feature_path / dependency_extractor.feature_name / dependency_extractor.get_label_file_name())
                return
            except Exception as e:
                print(str(e))
                print("didnt found dependency label file in {}".format(
                    self.feature_path / dependency_extractor.feature_name / dependency_extractor.get_label_file_name()))
                exit(-1)

    @classmethod
    def from_label_file(cls, label_file_path, out_path=FEATURES_DATA_PATH, source_path=RAW_DATA_PATH):
        """
        Initiate extractor from label file (csv) that contains
        filename and label columns.

        :param label_file_path:
        :return:
        """
        df = pd.read_csv(label_file_path)
        filenames = df['filename']
        labels = df['label']
        return cls(filenames, labels, out_path=out_path, source_path=source_path)

    @classmethod
    def magic_init(cls, feature_path=FEATURES_DATA_PATH, raw_path=RAW_DATA_PATH,
                   raw_label_filename='labels.csv'):
        """
        Initiate extractor deducting the paths by the extractor definition.

        :param label_file_path:
        :return:
        """

        from features import AVAILABLE_FEATURES
        out_path = feature_path
        if cls.dependency_feature_name:
            # source path is in feature path
            dependency_extractor = AVAILABLE_FEATURES[cls.dependency_feature_name]
            source_path = feature_path / dependency_extractor.feature_name
            label_file_name = dependency_extractor.get_label_file_name()
        else:
            # source path is raw data path
            source_path = raw_path
            label_file_name = raw_label_filename

        label_path = source_path / label_file_name
        print('info: read metadata from {}'.format(label_path))
        print('info: init extractor from {} to {}'.format(source_path, out_path))
        df = pd.read_csv(label_path)
        filenames = df['filename']
        labels = df['label']
        print('info: got filenames {}'.format(filenames))
        return cls(filenames, labels, out_path=out_path, source_path=source_path, feature_path=feature_path,
                   raw_path=raw_path)

    def remove_feature_files(self, feature_filenames=None):
        """
        Remove all files product from this extractor.
        :return:
        """
        if feature_filenames is None:
            feature_filenames = np.asarray(self.new_labels)[:, 0]
        [os.remove(self.out_path / filename) for filename in feature_filenames]
        os.remove(self.out_path / 'labels.{}.csv'.format(self.feature_name))

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
        input_path_warning_flag = True \
            if (self.dependency_feature_name and self.source_path != (
                    FEATURES_DATA_PATH / self.dependency_feature_name)) \
               or \
               (not self.dependency_feature_name and self.source_path != RAW_DATA_PATH) else False
        filename_format_warning_flag = True \
            if (self.dependency_feature_name and '.{}.npy'.format(self.dependency_feature_name) not in self.x[0]) \
               or \
               (not self.dependency_feature_name and self.x[0].split('.')[-1] not in AVAIL_MEDIA_TYPES) else False

        if self.dependency_feature_name:
            # """
            # If this parameter is given, the input is a feature in the Feature folder.
            # """
            if input_path_warning_flag:
                print("""warning: 
                {} Feature source folder is commonly FEATURES_DATA_PATH/{} config
                because it need {} feature as source, receeived {} instead.""".format(self.feature_name,
                                                                                      self.dependency_feature_name,
                                                                                      self.dependency_feature_name,
                                                                                      self.source_path))
            if filename_format_warning_flag:
                print("""warning:
                {} Feature source filenames are commonly formatted like <name>.{}.npy, received {} instead
                """.format(self.feature_name, self.dependency_feature_name, self.x[0]))
        else:
            # """
            # If this parameter isn't set, the raw_path should be the RAW_DATA_PATH,
            # also the files should end in an accepted format.
            # """
            if input_path_warning_flag:
                print('warning: this FeatureExtractor has a modified self.raw_path ({}), '
                      'but self.dependency_feature_name wasn\'t set. '.format(self.source_path))
                print('If this path doesn\t contain any audio files, '
                      'this extractor will probably fail.'
                      'Prefer to set RAW_DATA_PATH for using audio files.') if filename_format_warning_flag else None
            if filename_format_warning_flag:
                print('warning: No self.dependency_feature_name was set, and '
                      'the parsed filenames (self.x) has an unsupported media type ({}) '.format(
                    self.x[0].split('.')[-1]))
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
            if os.path.exists(self.source_path / x_i):
                y_i = self.y[i]
                new_x.append(x_i)
                new_y.append(y_i)
        self.x = new_x
        self.y = new_y

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, raw_path, **kwargs):
        def __process_element(data):
            """
            :param x: filename (str)
            :param y: label (str)
            :return:
            """
            # print('prosessing {}'.format(data))
            # x = data[0]
            # y = data[1]
            # # foo
            # product = 'foo'
            #
            # # this is kind-of standard
            # FeatureExtractor.save_feature(product, feature_name, out_path, x, y, new_labels)
            raise NotImplementedError()

        # stub
        return __process_element

    @staticmethod
    def process_elements(feature_name, new_labels, out_path, source_path, raw_path, fun=None, **kwargs):
        def __process_elements(data):
            for data_element in data:
                fun(data_element)

        return __process_elements

    def _parallel_transform(self, **kwargs):
        """
        Extract features in parallel.
        :param kwargs:
        :return:
        """
        self.clean_references()
        data = np.asarray([self.x, self.y]).swapaxes(0, 1)
        process_element = self.process_element(
            feature_name=self.feature_name,
            new_labels=self.new_labels,
            out_path=self.out_path,
            source_path=self.source_path,
            raw_path=self.raw_path,
            **kwargs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
            iterator = executor.map(process_element, data)
        list(iterator)
        self.export_new_labels()
        return np.asarray(self.new_labels)

    def _sequential_transform(self, **kwargs):
        """
        Extract features sequentially.
        :param kwargs:
        :return:
        """
        self.clean_references()
        data = np.asarray([self.x, self.y]).swapaxes(0, 1)
        print('info: starting sequential transform on data {}'.format(data))
        print('from {} to {}'.format(self.source_path, self.out_path))
        process_element = self.process_element(
            feature_name=self.feature_name,
            new_labels=self.new_labels,
            out_path=self.out_path,
            source_path=self.source_path,
            raw_path=self.raw_path,
            **kwargs)
        process_elements = self.process_elements(
            feature_name=self.feature_name,
            new_labels=self.new_labels,
            out_path=self.out_path,
            source_path=self.source_path,
            raw_path=self.raw_path,
            fun=process_element, **kwargs)
        process_elements(data)
        print('info: finished sequential transform, new labels are {}'.format(self.new_labels))
        self.export_new_labels()
        return np.asarray(self.new_labels)

    def transform(self, parallel=True, **kwargs):
        """
        Transform the data given in Labels to the inteded features.
        :param parallel:
        :param kwargs:
        :return:
        """
        if parallel:
            return self._parallel_transform(**kwargs)
        else:
            return self._sequential_transform(**kwargs)

    @classmethod
    def get_label_file_name(cls):
        return 'labels.{}.csv'.format(cls.feature_name)

    def get_label_file_path(self):
        return self.out_path / self.get_label_file_name()

    def export_new_labels(self):
        if not self.new_labels:
            # if new_labels is empty, then transform did nothing
            print('warning: {} did not process any data'.format(self))
            return
        df = pd.DataFrame(np.asarray(self.new_labels))
        df.columns = ['filename', 'label']
        df.to_csv(self.get_label_file_path(), index=False)

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


class MelSpectralCoefficientsFeatureExtractor(FeatureExtractor):
    feature_name = 'spec'

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, **kwargs):
        def __process_element(data):
            """
            :param x: filename (str)
            :param y: label (str)
            :return:
            """
            print('prosessing {}'.format(data))
            x = data[0]
            y = data[1]
            wav, _ = librosa.load(source_path / x, sr=SR)
            # Normalize audio signal
            wav = librosa.util.normalize(wav)
            # Get Mel-Spectrogram
            melspec = librosa.feature.melspectrogram(wav, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
                                                     fmin=FMIN,
                                                     fmax=FMAX, power=POWER)
            melspec = librosa.power_to_db(melspec).astype(np.float32)
            # this is kind-of standard
            FeatureExtractor.save_feature(melspec, feature_name, out_path, x, y, new_labels)

        return __process_element


# MFSC
class WindowedMelSpectralCoefficientsFeatureExtractor(FeatureExtractor):
    feature_name = 'windowed_spec'

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, **kwargs):
        def __process_element(data):
            """
            :param x: filename (str)
            :param y: label (str)
            :return:
            """
            print('prosessing {}'.format(data))
            x = data[0]
            y = data[1]

            # params

            # get song and split
            wav, _ = librosa.load(source_path / x, sr=SR)
            intervals = librosa.effects.split(
                # todo: split this extractor in two. One for this split, other for the windows
                wav,
                top_db=TOP_DB_WINDOWED_MFCC
            )
            # export intervals as new songs (wav)
            for interval_idx, interval in enumerate(intervals):
                if interval[1] - interval[0] < MIN_INTERVAL_LEN_WINDOWED_MFCC:
                    # if length is lesser that 1 second, discard interval
                    continue

                number_of_samples = wav.shape[0]
                number_of_windows = ceil(number_of_samples / WINDOW_LEN_WINDOWED_MFCC)
                for window_idx in range(number_of_windows):
                    start_idx = (window_idx * WINDOW_HOP_WINDOWED_MFCC)
                    end_idx = (start_idx + WINDOW_LEN_WINDOWED_MFCC)
                    window_wav = wav[start_idx:end_idx]

                    # Get Mel-Spectrogram
                    melspec = librosa.feature.melspectrogram(window_wav, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                             n_mels=N_MELS,
                                                             fmin=FMIN,
                                                             fmax=FMAX, power=POWER)
                    melspec = librosa.power_to_db(melspec).astype(np.float32)
                    # this is kind-of standard
                    filename = FeatureExtractor.get_file_name(x, feature_name,
                                                              ext='{}-{}.npy'.format(interval_idx, window_idx))
                    FeatureExtractor.save_feature(melspec, feature_name, out_path, x, y, new_labels, filename)

        return __process_element

    # def transform(self, parallel=False, **kwargs):
    #     if parallel:
    #         raise Exception('error: {} cannot be ran in paralel'.format(self.feature_name))
    #     return super().transform(parallel, **kwargs)


# Singing Voice Detection Pipeline
class DoubleHPSSFeatureExtractor(FeatureExtractor):
    feature_name = '2hpss'

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, **kwargs):
        """
        Wrapper for actual function __process_elements(data)
        :param feature_name:
        :param new_labels:
        :param out_path:
        :param source_path:
        :param fun:
        :param model_name:
        :param kwargs:
        :return:
        """
        def __process_element(data):
            """
            Compute double stage HPSS for the given audio file
            extracted from https://github.com/kyungyunlee/ismir2018-revisiting-svd/blob/master/leglaive_lstm/audio_processor.py
            :param x: filename (str)
            :param y: label (str)
            :return: mel_D2_total : concatenated melspectrogram of percussive, harmonic components of double stage HPSS. Shape=(2 * n_bins, total_frames) ex. (80, 2004)
            """
            print('processing {}'.format(data))
            x_i = data[0]
            y_i = data[1]

            file_name = FeatureExtractor.get_file_name(x_i, feature_name)
            try:
                # try to load if file already exist
                np.load(out_path / file_name, allow_pickle=True)
                print('info: {} loaded from .npy !'.format(file_name))
                new_labels.append([file_name, y_i])
            except FileNotFoundError or OSError or EOFError:
                # OSError and EOFError are raised if file are inconsistent
                audio_src, _ = librosa.load(source_path / x_i, sr=SR_HPSS)
                # Normalize audio signal
                audio_src = librosa.util.normalize(audio_src)
                # first HPSS
                D_harmonic, D_percussive = ono_hpss(audio_src, N_FFT_HPSS_1, N_HOP_HPSS_1)
                # second HPSS
                D2_harmonic, D2_percussive = ono_hpss(D_percussive, N_FFT_HPSS_2, N_HOP_HPSS_2)

                # compute melgram
                mel_harmonic = log_melgram(D2_harmonic, SR_HPSS, N_FFT_HPSS_2, N_HOP_HPSS_2, N_MELS_HPSS)
                mel_percussive = log_melgram(D2_percussive, SR_HPSS, N_FFT_HPSS_2, N_HOP_HPSS_2, N_MELS_HPSS)
                # concat
                mel_total = np.vstack((mel_harmonic, mel_percussive))

                # this is kind-of standard
                FeatureExtractor.save_feature(mel_total, feature_name, out_path, x_i, y_i, new_labels)

        return __process_element

class VoiceActivationFeatureExtractor(FeatureExtractor):
    feature_name = 'voice_activation'
    dependency_feature_name = DoubleHPSSFeatureExtractor.feature_name

    @staticmethod
    def frame_level_predict(y_pred, number_of_mel_samples):
        """
        Predict Voice Activity Regions at a Frame Level for a given song.
        For each frame of the MFCC a Voice Detection Probability is predicted, then the output have shape: (n_frames, 1)

        :param model_name: name of the trained model
        :param filename:  path to the music file to be predicted
        :param cache: flag to optimize heavy operations with caching in disk
        :param plot: flag to plot MFCCs and SVD in an aligned plot if GUI available.
        :return: (Time, Predictions): SVD probabilities at frame level with time markings
        """
        # transform raw predictions to frame level
        aligned_y_pred = [[] for _ in range(number_of_mel_samples)]
        for first_frame_idx, window_prediction in enumerate(y_pred):
            # for each prediction
            for offset, frame_prediction in enumerate(window_prediction):
                # accumulate overlapped predictions in a list
                aligned_y_pred[first_frame_idx + offset].append(frame_prediction[0])

        # frame_level_y_pred = []
        # for _, predictions in enumerate(aligned_y_pred[:-1]):
        #     # -1 because last element is empty
        #     # reduce the overlapped predictions to a single value
        #     frame_level_y_pred.append(min(predictions))

        time = frames_to_time(range(number_of_mel_samples), sr=SR_HPSS, n_fft=N_FFT_HPSS_2,
                              hop_length=N_HOP_HPSS_2)

        print('info: done')
        return time, aligned_y_pred

    @staticmethod
    def post_process(input, number_of_mel_samples):
        """
        Export in a time-dependant domain instead of sample-domain; as sample rate change between methods.
        :param input:
        :return:
        """
        return np.asarray(VoiceActivationFeatureExtractor.frame_level_predict(input, number_of_mel_samples))

    @staticmethod
    def process_elements(feature_name, new_labels, out_path, source_path, fun=None,
                         model_name=VOICE_DETECTION_MODEL_NAME,
                         **kwargs):
        """
        Wrapper for actual function __process_elements(data)
        :param feature_name:
        :param new_labels:
        :param out_path:
        :param source_path:
        :param fun:
        :param model_name:
        :param kwargs:
        :return:
        """
        def __process_elements(data):
            """
            :param data: shape (#_songs, 2) the axis 1 corresponds to the filename/label pair
            :return:
            """
            x = data[:, 0]
            y = data[:, 1]
            print('loaded metadata in {}'.format(data))

            from keras.models import load_model
            from keras import backend

            if len(backend.tensorflow_backend._get_available_gpus()) > 0:
                # set gpu number
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            # load mode
            loaded_model = load_model(str(MODELS_DATA_PATH / 'leglaive' / 'rnn_{}.h5'.format(model_name)))
            print("loaded model")
            print(loaded_model.summary())

            mean_std = np.load(MODELS_DATA_PATH / 'leglaive' / 'train_mean_std_{}.npy'.format(model_name))
            mean = mean_std[0]
            std = mean_std[1]

            for idx, x_i in enumerate(x):
                # this is kind-of standard
                y_i = y[idx]
                file_name = FeatureExtractor.get_file_name(x_i, feature_name)
                try:
                    # try to load if file already exist
                    np.load(out_path / file_name, allow_pickle=True)
                    print('info: {} loaded from .npy !'.format(file_name))
                    new_labels.append([file_name, y_i])
                except FileNotFoundError or OSError or EOFError:
                    # OSError and EOFError are raised if file are inconsistent
                    # final_shape: (#_hops, #_mel_filters, #_window)
                    print('info: loading hpss data for {}'.format(x_i))
                    hpss = np.load(source_path / x_i)  # _data, #_coefs, #_samples)
                    print('info: formatting data')
                    try:
                        padding = RNN_INPUT_SIZE_VOICE_ACTIVATION - hpss.shape[1]
                        if padding > 0:
                            # if hpss is shorter that RNN input shape, then add padding on axis=1
                            hpss = np.pad(hpss, ((0, 0), (0, padding)), mode='constant')
                        number_of_mel_samples = hpss.shape[1]
                        # at least should have 1 window
                        number_of_steps = max(number_of_mel_samples - RNN_INPUT_SIZE_VOICE_ACTIVATION, 1)
                        total_x = np.array([hpss[:, i: i + RNN_INPUT_SIZE_VOICE_ACTIVATION]
                                            for i in range(0, number_of_steps, 1)])
                        # final_shape: (#_hops, #_mel_filters, #_window)

                        total_x_norm = (total_x - mean) / std
                        total_x_norm = np.swapaxes(total_x_norm, 1, 2)
                        # final_shape: (#_hops, #_window, #_mel_filters)

                        x_test = total_x_norm
                        print('info: predicting')
                        y_pred = loaded_model.predict(x_test, verbose=1)  # Shape=(total_frames,)
                        time, aligned_y_pred = VoiceActivationFeatureExtractor.post_process(y_pred, number_of_mel_samples)
                        print('info: predicted!')
                        result_array = np.asarray([time, aligned_y_pred])
                        FeatureExtractor.save_feature(result_array, feature_name, out_path, x_i, y_i, new_labels)
                    except MemoryError as e:
                        print('error: memory error while proccessing {}. Ignoring...'.format(x_i))
                        print(e)

        return __process_elements

    def transform(self, parallel=False, **kwargs):
        if parallel:
            raise Exception('error: {} cannot be ran in paralel'.format(self.feature_name))
        return super().transform(parallel, **kwargs)


class MeanSVDFeatureExtractor(FeatureExtractor):
    feature_name = 'mean_svd'
    dependency_feature_name = VoiceActivationFeatureExtractor.feature_name

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, **kwargs):
        def __process_element(data):
            """
            Flatten overlapped prediction by Leglaive SVD prediction by calculating mean for every frame.
            """
            print('prosessing {}'.format(data))
            x_i = data[0]
            y_i = data[1]

            file_name = FeatureExtractor.get_file_name(x_i, feature_name)
            try:
                # try to load if file already exist
                np.load(out_path / file_name, allow_pickle=True)
                print('info: {} loaded from .npy !'.format(file_name))
                new_labels.append([file_name, y_i])
            except FileNotFoundError or OSError or EOFError:
                # OSError and EOFError are raised if file are inconsistent
                voice_activation = np.load(source_path / x_i, allow_pickle=True)
                mean_voice_activation = np.asarray([np.mean(elem) for elem in voice_activation[
                    1]])  # calculate mean for each frame predictions (~218 per frame)
                mean_voice_activation = np.nan_to_num(
                    mean_voice_activation)  # remove NaNs product of mean of empty frame
                # this is kind-of standard
                FeatureExtractor.save_feature([voice_activation[0], mean_voice_activation], feature_name, out_path, x_i,
                                              y_i, new_labels)

        return __process_element


class SVDPonderatedVolumeFeatureExtractor(FeatureExtractor):
    feature_name = 'svd_ponderated_volume'
    dependency_feature_name = MeanSVDFeatureExtractor.feature_name

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, raw_path, **kwargs):
        def __process_element(data):
            """

            """
            print('prosessing {}'.format(data))
            x_i = data[0]
            y_i = data[1]

            file_name = FeatureExtractor.get_file_name(x_i, feature_name, 'wav')
            try:
                # try to load if file already exist
                librosa.load(out_path / file_name, sr=SR)
                print('info: {} loaded from .npy !'.format(file_name))
                new_labels.append([file_name, y_i])
            except FileNotFoundError or OSError or EOFError:
                # OSError and EOFError are raised if file are inconsistent
                mean_voice_activation = np.load(source_path / x_i, allow_pickle=True)
                audio_src, _ = librosa.load(
                    raw_path / '{}.mp3'.format(x_i.split('.{}'.format(DoubleHPSSFeatureExtractor.feature_name))[0]),
                    sr=SR)  # todo: support other formats

                time = mean_voice_activation[0]
                voice_activation_prob = mean_voice_activation[1]

                last_sample = 0
                for idx, time_tick in enumerate(time):
                    voice_prob = voice_activation_prob[idx]
                    init_sample = last_sample
                    last_sample = int(time_tick * SR)
                    audio_src[init_sample:last_sample] *= voice_prob
                # this is kind-of standard
                FeatureExtractor.save_audio(audio_src, feature_name, out_path, x_i, y_i, new_labels)

        return __process_element


class IntensitySplitterFeatureExtractor(FeatureExtractor):
    feature_name = 'intensity_split'

    @staticmethod
    def process_element(feature_name, new_labels, out_path, source_path, **kwargs):
        def __process_element(data):
            """
            :param x: filename (str)
            :param y: label (str)
            :return:
            """
            print('prosessing {}'.format(data))
            x = data[0]
            y = data[1]

            # params

            # get song and split
            wav, _ = librosa.load(source_path / x, sr=SR)
            intervals = librosa.effects.split(
                # todo: split this extractor in two. One for this split, other for the windows
                wav,
                top_db=TOP_DB_WINDOWED_MFCC
            )
            # export intervals as new songs (wav)
            for interval_idx, interval in enumerate(intervals):
                if interval[1] - interval[0] < MIN_INTERVAL_LEN_WINDOWED_MFCC:
                    # if length is lesser that 1 second, discard interval
                    continue
                FeatureExtractor.save_audio(wav[interval[0]:interval[1]], feature_name, out_path, x, y, new_labels)

        return __process_element

    # def transform(self, parallel=False, **kwargs):
    #     if parallel:
    #         raise Exception('error: {} cannot be ran in paralel'.format(self.feature_name))
    #     return super().transform(parallel, **kwargs)


AVAILABLE_FEATURES = {MelSpectralCoefficientsFeatureExtractor.feature_name: MelSpectralCoefficientsFeatureExtractor,
                      WindowedMelSpectralCoefficientsFeatureExtractor.feature_name: WindowedMelSpectralCoefficientsFeatureExtractor,
                      DoubleHPSSFeatureExtractor.feature_name: DoubleHPSSFeatureExtractor,
                      VoiceActivationFeatureExtractor.feature_name: VoiceActivationFeatureExtractor,
                      MeanSVDFeatureExtractor.feature_name: MeanSVDFeatureExtractor,
                      SVDPonderatedVolumeFeatureExtractor.feature_name: SVDPonderatedVolumeFeatureExtractor}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--raw_path', help='Source path where audio data files are stored', default=RAW_DATA_PATH)
    parser.add_argument('--features_path', help='Output path where exported data will be placed', default=FEATURES_DATA_PATH)
    # parser.add_argument('--label_filename', help='Source path where label file is stored', default='labels.csv')
    parser.add_argument('--feature', help='name of the feature to be extracted (options: mfsc, leglaive)',
                        default='windowed_spec')

    args = parser.parse_args()
    raw_path = pathlib.Path(args.raw_path)
    features_path = pathlib.Path(args.features_path)
    # label_path = raw_path / args.label_filename
    feature_name = args.feature
    print('info: from {} to {}'.format(raw_path, features_path))
    extractor = AVAILABLE_FEATURES[feature_name]
    extractor = extractor.magic_init(feature_path=features_path, raw_path=raw_path)
    extractor.transform()
