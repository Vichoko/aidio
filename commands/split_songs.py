"""
Command to split MP3/WAV audio dataset in smaller binary chunks and optional mfcc.

…unks for easier loading.
"""
import argparse
import concurrent.futures
import inspect
import os
import pathlib
import sys
from math import ceil

import librosa
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import SR, MFCC_N_COEF, MFCC_FFT_WINDOW, MFCC_HOP_LENGTH, FEATURE_EXTRACTOR_NUM_WORKERS, MFCC_N_MELS

MAX_CLASS_NUMBER = 0  # Number of classes; 0 is all possible
SPLIT_AUDIO_LENGTH = 11  # Second
OUTPUT = '2d'
SEQUENTIAL = True


def make_handler(new_labels, new_filenames, data_path, out_path):
    def handler(filename, label):
        wav, sr = librosa.load(
            str(data_path / filename),
            sr=SR
        )
        # channels = wav.shape[0]
        number_of_samples = wav.shape[0]
        extension = filename.split('.')[-1]
        slice_number_of_samples = SPLIT_AUDIO_LENGTH * sr
        number_of_slices = ceil(number_of_samples / slice_number_of_samples)
        for slice_idx in range(number_of_slices):
            first_sample = slice_idx * slice_number_of_samples
            last_sample = (slice_idx + 1) * slice_number_of_samples
            new_wav = wav[first_sample: last_sample]
            if new_wav.shape[0] < slice_number_of_samples:
                # if new slice is short, discard it to avoid issues
                print('warning: Slice {} of {} discarded because was too short.'.format(slice_idx, filename))
                continue
            if OUTPUT == '1d':
                new_filename = filename.replace(extension, '{}.npy'.format(slice_idx))
                data = new_wav
            elif OUTPUT == '2d':
                new_filename = filename.replace(extension, 'mfcc.{}.npy'.format(slice_idx))
                if os.path.isfile(out_path / new_filename):
                    print('{} already exists skipping...'.format(new_filename))
                    new_filenames.append(new_filename)
                    new_labels.append(label)
                    continue
                # Normalize audio signal
                new_wav = librosa.util.normalize(new_wav)
                # Get Mel-Spectrogram
                mfcc = librosa.feature.mfcc(new_wav, sr=SR, n_mfcc=MFCC_N_COEF, n_fft=MFCC_FFT_WINDOW,
                                            hop_length=MFCC_HOP_LENGTH, n_mels=MFCC_N_MELS)
                data = mfcc
            else:
                raise NotImplementedError()
            np.save(
                out_path / new_filename,
                data
            )
            print('info: {} exported successfully.'.format(new_filename))
            new_filenames.append(new_filename)
            new_labels.append(label)
        return
    return handler


def main():
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        help='Path to the folder where the songs are stored')
    parser.add_argument('out_path',
                        help='Path to folder where data is going to be stored.')
    parser.add_argument('--label_filename',
                        help='file name of label file where filename and labels are stored.',
                        default='labels.csv')
    args = parser.parse_args()
    data_path = pathlib.Path(args.data_path)
    label_filename = args.label_filename
    out_path = pathlib.Path(args.out_path)

    if data_path == out_path:
        raise Exception('data_path cannot be same as out_path')

    df = pd.read_csv(data_path / label_filename)
    filenames = df['filename']
    labels = df['label']
    new_filenames = []
    new_labels = []
    # class_set = set()
    assert len(filenames) == len(labels)
    handler = make_handler(new_labels, new_filenames, data_path, out_path)
    if SEQUENTIAL:
        for idx, filename in enumerate(filenames):
            label = labels[idx]
            handler(filename, label)
            print('debug: new_labels has {} elements.'.format(len(new_labels))) if debug else None
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=FEATURE_EXTRACTOR_NUM_WORKERS) as executor:
            iterator = executor.map(handler, filenames, labels)
        list(iterator)  # wait to finish
    df = pd.DataFrame(np.asarray([new_filenames, new_labels]).swapaxes(0, 1))
    df.columns = ['filename', 'label']
    df.to_csv(out_path / label_filename, index=False)
    print('info: New {} exported successfully.'.format(label_filename))


if __name__ == '__main__':
    main()
