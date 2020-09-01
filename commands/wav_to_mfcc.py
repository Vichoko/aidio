"""
Copy the sound and label files fron a dir to antoher based on given labelfile
"""
import argparse
import inspect
# empirically defined 20 MB .mp3 file is big enough to crash feature extractors
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import NUMBER_OF_CLASSES, SR, MFCC_N_COEF, MFCC_FFT_WINDOW, MFCC_HOP_LENGTH, MFCC_N_MELS


def load_csv(path, filename):
    # load source data from csv
    df = pd.read_csv(path / filename)
    filenames = np.asarray(df['filename'])
    labels = np.asarray(df['label'])
    return filenames, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--src_path', help='Parent directory path where source label files are stored', )
    parser.add_argument('--dest_path', help='Directory path where label files are going to be stored', )
    parser.add_argument('--src_label_prefix', help='file name of feature label file', default='labels')
    # arg parsing
    args = parser.parse_args()
    src_path = Path(args.src_path)
    dest_path = Path(args.dest_path)
    src_label_prefix = Path(args.src_label_prefix)

    set_names = ['train', 'test', 'val']
    for set_name in set_names:
        # load source metadata from csv
        # copy label file
        label_filename = '{}.{}.{}.csv'.format(src_label_prefix, NUMBER_OF_CLASSES, set_name)
        # copy files
        wav_filenames, labels = load_csv(src_path, label_filename)
        mfcc_filenames = []
        for data_idx, wav_filename in enumerate(wav_filenames):
            # load and process wav to mfcc
            wav = np.load(src_path / wav_filename)
            mfcc = librosa.feature.mfcc(
                wav,
                sr=SR,
                n_mfcc=MFCC_N_COEF,
                n_fft=MFCC_FFT_WINDOW,
                hop_length=MFCC_HOP_LENGTH,
                n_mels=MFCC_N_MELS
            )
            mfcc_filename = wav_filename.replace('.npy', 'mfcc.npy')
            mfcc_filenames.append(mfcc_filename)
            np.save(
                dest_path / mfcc_filename,
                mfcc
            )
            print('info: {} exported successfully.'.format(mfcc_filename))
        # export labels
        df = pd.DataFrame(np.asarray([mfcc_filenames, labels]).swapaxes(0, 1))
        df.columns = ['filename', 'label']
        df.to_csv(dest_path / label_filename, index=False)
        print('info: New {} exported successfully.'.format(label_filename))
