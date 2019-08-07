from _ast import Not

import librosa
import sys

from config import SR, RAW_DATA_PATH, FEATURES_DATA_PATH, HOP_LENGTH, N_FFT, N_MELS, FMIN, FMAX, POWER, \
    VOICE_DETECTION_PATH, VOICE_DETECTION_MODEL_NAME
from interfaces import FeatureExtractor
import argparse
import pathlib
import numpy as np


#
# def process_single_audio_double_hpss(audio_file):
#     ''' Compute double stage HPSS for the given audio file
#     Args :
#         audio_file : path to audio file
#     Return :
#         mel_D2_total : concatenated melspectrogram of percussive, harmonic components of double stage HPSS. Shape=(2 * n_bins, total_frames) ex. (80, 2004)
#     '''
#
#     audio_src, _ = librosa.load(audio_file, sr=SR)
#     # Normalize audio signal
#     audio_src = librosa.util.normalize(audio_src)
#     # first HPSS
#     D_harmonic, D_percussive = ono_hpss(audio_src, N_FFT1, N_HOP1)
#     # second HPSS
#     D2_harmonic, D2_percussive = ono_hpss(D_percussive, N_FFT2, N_HOP2)
#
#     assert D2_harmonic.shape == D2_percussive.shape
#     print(D2_harmonic.shape, D2_percussive.shape)
#
#     # compute melgram
#     mel_harmonic = log_melgram(D2_harmonic, SR, N_FFT2, N_HOP2, N_MELS)
#     mel_percussive = log_melgram(D2_percussive, SR, N_FFT2, N_HOP2, N_MELS)
#     # concat
#     mel_total = np.vstack((mel_harmonic, mel_percussive))
#
#     print(mel_total.shape)
#     return mel_total


class MFCCFeatureExtractor(FeatureExtractor):
    feature_name = 'spec'

    @staticmethod
    def process_element(feature_name, new_labels, out_path):
        def __process_element(data):
            """
            :param x: filename (str)
            :param y: label (str)
            :return:
            """
            print('prosessing {}'.format(data))
            x = data[0]
            y = data[1]
            wav, _ = librosa.load(RAW_DATA_PATH / x, sr=SR)
            # Normalize audio signal
            wav = librosa.util.normalize(wav)
            # Get Mel-Spectrogram
            melspec = librosa.feature.melspectrogram(wav, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
                                                     fmin=FMIN,
                                                     fmax=FMAX, power=POWER)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            # this is kind-of standard
            name = '.'.join(x.split('.')[:-1])
            filename = '{}.{}.npy'.format(name, feature_name)
            np.save(out_path / filename, melspec)
            new_labels.append([filename, y])
            print('info: {} transformed and saved!'.format(filename))

        return __process_element

    def parallel_transform(self, **kwargs):
        return super(MFCCFeatureExtractor, self).parallel_transform(feature_name=self.feature_name,
                                                                    out_path=self.out_path,
                                                                    new_labels=self.new_labels)


class VoiceActivationFeatureExtractor(FeatureExtractor):
    feature_name = 'voice_activation'

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
            sys.path.append(VOICE_DETECTION_PATH)
            print('info: importing predictor')
            from leglaive_lstm import frame_level_predict
            print('info: predictor imported succesfully!')

            try:
                x_pred, y_pred = frame_level_predict(VOICE_DETECTION_MODEL_NAME, raw_path / x, cache=True)
            except FileNotFoundError as e:
                if x in str(e):
                    return
                raise e

            print('info: predicted !')
            pred = np.asarray([x_pred, y_pred])

            # this is kind-of standard
            name = '.'.join(x.split('.')[:-1])
            filename = '{}.{}.npy'.format(name, feature_name)
            np.save(out_path / filename, pred)
            new_labels.append([filename, y])
            print('info: {} transformed and saved!'.format(filename))
        return __process_element

    def parallel_transform(self, **kwargs):
        return super(VoiceActivationFeatureExtractor, self).parallel_transform(feature_name=self.feature_name,
                                                                               out_path=self.out_path,
                                                                               new_labels=self.new_labels,
                                                                               raw_path=self.raw_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--source_path', help='Source path where data files is stored', default=RAW_DATA_PATH)
    parser.add_argument('--out_path', help='Output path where exported data will be placed', default=FEATURES_DATA_PATH)
    parser.add_argument('--label_filename', help='Source path where label file is stored', default='labels.csv')
    parser.add_argument('--feature', help='name of the feature to be extracted (options: mfsc, leglaive)', default='leglaive')

    args = parser.parse_args()
    source_path = pathlib.Path(args.source_path)
    out_path = pathlib.Path(args.out_path)
    label_path = source_path / args.label_filename
    feature_name = args.feature

    print('info: from {} to {}'.format(source_path, out_path))
    if feature_name == 'leglaive':
        extractor = VoiceActivationFeatureExtractor.from_label_file(label_path, out_path=out_path, raw_path=source_path)
    elif feature_name == 'mfsc':
        extractor = MFCCFeatureExtractor.from_label_file(label_path, out_path=out_path, raw_path=source_path)
    else:
        raise NotImplemented('{} feature not implemented'.format(feature_name))
    extractor.parallel_transform()
