"""
Transform label file(s) from a feature to another.
Supports mainly already splited label files.
"""
import argparse
import inspect
# empirically defined 20 MB .mp3 file is big enough to crash feature extractors
import os
import sys
from pathlib import Path

import numpy as np

from config import NUMBER_OF_CLASSES, RANDOM_SEED

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd


# from config import *

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
    parser.add_argument('--dest_label_prefix', help='file name of feature label file', default='labels.mfcc')
    # arg parsing
    args = parser.parse_args()
    src_path = Path(args.src_path)
    dest_path = Path(args.dest_path)
    src_label_prefix = Path(args.src_label_prefix)
    dest_label_prefix = Path(args.dest_label_prefix)

    # real logic
    # open source already splitted label files according to the format: <label_prefix>.<n_classes>.<set_name>.csv
    set_names = ['train', 'test', 'val']

    for set_name in set_names:
        # load source metadata from csv
        label_filename = '{}.{}.{}.csv'.format(src_label_prefix, NUMBER_OF_CLASSES, set_name)
        filenames, labels = load_csv(src_path, label_filename)
        # get unique song names from filenames
        songs = set()
        for data_idx, filename in enumerate(filenames):
            label = labels[data_idx]
            song_name = filename.split('.')[0]
            songs.add(song_name)

        # load dest metadata from csv
        label_filename = '{}.csv'.format(dest_label_prefix)
        filenames, labels = load_csv(dest_path, label_filename)
        # shuffle data elements
        indices = np.arange(len(filenames))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
        filenames = np.asarray(filenames[indices])
        labels = np.asarray(labels[indices])
        # filter only the songs in the source metadata
        out_indices = []
        for data_idx, filename in enumerate(filenames):
            label = labels[data_idx]
            song_name = filename.split('.')[0]
            if song_name in songs:
                # if song name is in selected song set, append the index for further filtering
                out_indices.append(data_idx)
        # export filtered filenames and labels to CSV
        label_filename = '{}.{}.{}.csv'.format(dest_label_prefix, NUMBER_OF_CLASSES, set_name)
        pd.DataFrame(
            {
                'filename': filenames[out_indices],
                'label': labels[out_indices],
            }
        ).to_csv(dest_path / label_filename)
