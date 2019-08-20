import os
import librosa

from librosa import frames_to_time

from config import SR, RAW_DATA_PATH, FEATURES_DATA_PATH, HOP_LENGTH, N_FFT, N_MELS, FMIN, FMAX, POWER, \
    VOICE_DETECTION_MODEL_NAME, N_FFT_HPSS_1, N_HOP_HPSS_1, N_FFT_HPSS_2, N_HOP_HPSS_2, SR_HPSS, \
    N_MELS_HPSS, MODELS_DATA_PATH, RNN_INPUT_SIZE_VOICE_ACTIVATION
from interfaces import FeatureExtractor
import argparse
import pathlib
import numpy as np

from util.leglaive.audio import ono_hpss, log_melgram


class MelSpectralCoefficientsFeatureExtractor(FeatureExtractor):
    feature_name = 'spec'

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
            wav, _ = librosa.load(raw_path / x, sr=SR)
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

    def parallel_transform(self, **kwargs):
        # the parameters of parallel_transform are passed to process_element and process_elements methods
        return super(MelSpectralCoefficientsFeatureExtractor, self).parallel_transform()


# Singing Voice Detection Pipeline
class DoubleHPSSFeatureExtractor(FeatureExtractor):
    feature_name = '2hpss'

    @staticmethod
    def process_element(feature_name, new_labels, out_path, raw_path, **kwargs):
        def __process_element(data):
            """
            Compute double stage HPSS for the given audio file
            extracted from https://github.com/kyungyunlee/ismir2018-revisiting-svd/blob/master/leglaive_lstm/audio_processor.py
            :param x: filename (str)
            :param y: label (str)
            :return: mel_D2_total : concatenated melspectrogram of percussive, harmonic components of double stage HPSS. Shape=(2 * n_bins, total_frames) ex. (80, 2004)
            """
            print('prosessing {}'.format(data))
            x_i = data[0]
            y_i = data[1]

            audio_src, _ = librosa.load(raw_path / x_i, sr=SR_HPSS)
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
    def proccess_elements(feature_name, new_labels, out_path, raw_path, fun=None, model_name=VOICE_DETECTION_MODEL_NAME,
                          **kwargs):
        def __process_elements(data):
            """
            :param data: shape (#_songs, 2) the axis 1 corresponds to the filename/label pair
            :return:
            """
            x = data[:, 0]
            y = data[:, 1]

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
                _, file_name = FeatureExtractor.get_file_name(x_i, feature_name)
                try:
                    # try to load if file already exist
                    np.load(out_path / file_name, allow_pickle=True)
                    print('info: {} loaded from .npy !'.format(file_name))
                    new_labels.append([file_name, y_i])
                except FileNotFoundError or OSError or EOFError:
                    # OSError and EOFError are raised if file are inconsistent
                    hpss = np.load(raw_path / x_i)  # _data, #_coefs, #_samples)

                    number_of_mel_samples = hpss.shape[1]
                    total_x = np.array([hpss[:, i: i + RNN_INPUT_SIZE_VOICE_ACTIVATION]
                                        for i in range(0, number_of_mel_samples - RNN_INPUT_SIZE_VOICE_ACTIVATION, 1)])
                    # final_shape: (#_hops, #_mel_filters, #_window)

                    total_x_norm = (total_x - mean) / std
                    total_x_norm = np.swapaxes(total_x_norm, 1, 2)
                    # final_shape: (#_hops, #_window, #_mel_filters)

                    x_test = total_x_norm
                    y_pred = loaded_model.predict(x_test, verbose=1)  # Shape=(total_frames,)
                    time, aligned_y_pred = VoiceActivationFeatureExtractor.post_process(y_pred, number_of_mel_samples)
                    print('info: predicted!')
                    result_array = np.asarray([time, aligned_y_pred])
                    FeatureExtractor.save_feature(result_array, feature_name, out_path, x_i, y_i, new_labels)

        return __process_elements

    def parallel_transform(self, **kwargs):
        return super(VoiceActivationFeatureExtractor, self).parallel_transform(parallel=False)


class MeanSVDFeatureExtractor(FeatureExtractor):
    feature_name = 'mean_svd'
    dependency_feature_name = VoiceActivationFeatureExtractor.feature_name

    @staticmethod
    def process_element(feature_name, new_labels, out_path, raw_path, **kwargs):
        def __process_element(data):
            """
            Flatten overlapped prediction by Leglaive SVD prediction by calculating mean for every frame.
            """
            print('prosessing {}'.format(data))
            x_i = data[0]
            y_i = data[1]

            voice_activation = np.load(raw_path / x_i, allow_pickle=True)
            mean_voice_activation = np.asarray([np.mean(elem) for elem in voice_activation[
                1]])  # calculate mean for each frame predictions (~218 per frame)
            mean_voice_activation = np.nan_to_num(mean_voice_activation)  # remove NaNs product of mean of empty frame
            # this is kind-of standard
            FeatureExtractor.save_feature([voice_activation[0], mean_voice_activation], feature_name, out_path, x_i,
                                          y_i, new_labels)

        return __process_element

    def parallel_transform(self, **kwargs):
        return super(MeanSVDFeatureExtractor, self).parallel_transform(parallel=True)


class SVDPonderatedVolumeFeatureExtractor(FeatureExtractor):
    feature_name = 'svd_ponderated_volume'
    dependency_feature_name = MeanSVDFeatureExtractor.feature_name

    @staticmethod
    def process_element(feature_name, new_labels, out_path, raw_path, **kwargs):
        def __process_element(data):
            """

            """
            print('prosessing {}'.format(data))
            x_i = data[0]
            y_i = data[1]

            mean_voice_activation = np.load(raw_path / x_i, allow_pickle=True)
            audio_src, _ = librosa.load(RAW_DATA_PATH / '{}.mp3'.format(x_i.split('.')[0]),
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

    def parallel_transform(self, **kwargs):
        return super(SVDPonderatedVolumeFeatureExtractor, self).parallel_transform(parallel=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--source_path', help='Source path where data files is stored', default=RAW_DATA_PATH)
    parser.add_argument('--out_path', help='Output path where exported data will be placed', default=FEATURES_DATA_PATH)
    parser.add_argument('--label_filename', help='Source path where label file is stored', default='labels.csv')
    parser.add_argument('--feature', help='name of the feature to be extracted (options: mfsc, leglaive)',
                        default='svd_ponderated_volume')

    args = parser.parse_args()
    source_path = pathlib.Path(args.source_path)
    out_path = pathlib.Path(args.out_path)
    label_path = source_path / args.label_filename
    feature_name = args.feature

    print('info: from {} to {}'.format(source_path, out_path))
    if feature_name == 'leglaive':
        # leglaive use hpss as input
        src_feature_name = DoubleHPSSFeatureExtractor.feature_name
        source_path = FEATURES_DATA_PATH
        label_path = source_path / src_feature_name / 'labels.{}.csv'.format(src_feature_name)
        extractor = VoiceActivationFeatureExtractor.from_label_file(label_path, out_path=out_path, raw_path=source_path)
    elif feature_name == 'mfsc':
        extractor = MelSpectralCoefficientsFeatureExtractor.from_label_file(label_path, out_path=out_path,
                                                                            raw_path=source_path)
    elif feature_name == '2hpss':
        extractor = DoubleHPSSFeatureExtractor.from_label_file(label_path, out_path=out_path, raw_path=source_path)
    elif feature_name == 'svd_ponderated_volume':
        # leglaive use SVD as input
        src_feature_name = MeanSVDFeatureExtractor.feature_name
        source_path = FEATURES_DATA_PATH / src_feature_name
        label_path = source_path / 'labels.{}.csv'.format(src_feature_name)
        extractor = SVDPonderatedVolumeFeatureExtractor.from_label_file(label_path, out_path=out_path,
                                                                        raw_path=source_path)
    elif feature_name == 'mean_svd':
        # leglaive use SVD as input
        # todo: unify this behaviour
        src_feature_name = VoiceActivationFeatureExtractor.feature_name
        source_path = FEATURES_DATA_PATH / src_feature_name
        label_path = source_path / 'labels.{}.csv'.format(src_feature_name)
        extractor = MeanSVDFeatureExtractor.from_label_file(label_path, out_path=out_path,
                                                            raw_path=source_path)
    else:
        raise NotImplemented('{} feature not implemented'.format(feature_name))
    extractor.parallel_transform()
