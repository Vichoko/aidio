"""
Transform label file(s) from a feature to another.
Supports mainly already splited label files.
"""
import argparse
import inspect
# empirically defined 20 MB .mp3 file is big enough to crash feature extractors
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import NUMBER_OF_CLASSES

import pandas as pd


# from config import *

def load_csv(path, filename):
    # load source data from csv
    df = pd.read_csv(path / filename)
    filenames = np.asarray(df['filename'])
    labels = np.asarray(df['label'])
    return filenames, labels


def compare_filenames(filenames1, filenames2):
    # get unique song names from filenames
    songs1 = set()
    sets1 = defaultdict(lambda: defaultdict(set))
    for data_idx, filename in enumerate(filenames1):
        if filename:
            song_name = filename.split('.')[0]
            songs1.add(song_name)
            sets1[song_name]['pieces'].add(filename.split('.')[-2])
    songs2 = set()
    sets2 = defaultdict(lambda: defaultdict(set))
    for data_idx, filename in enumerate(filenames2):
        if filename:
            song_name = filename.split('.')[0]
            songs2.add(song_name)
            sets2[song_name]['pieces'].add(filename.split('.')[-2])
    comparative_conditionals = [
        (len(songs1), len(songs2), 'COUNT TEST'),
        (songs1.intersection(songs2), songs1, 'EQUALITY TEST 1'),
        (songs2.intersection(songs1), songs2, 'EQUALITY TEST 2'),
        (songs1, songs2, 'EQUALITY TEST 3'),
    ]

    def equal_comparator(conditionals, verbose=True):
        for idx, conditional in enumerate(conditionals):
            cond1 = conditional[0]
            cond2 = conditional[1]
            title = conditionals[2]
            value = cond1 == cond2
            if verbose:
                print('info: {}'.format(title))
                print('info []: check!').format(idx) if value else print(
                    'warning []: FAILED. {} != {}.'.format(cond1, cond2))
            yield value

    print('info: Song tests')
    equal_comparator(comparative_conditionals)
    # here we assume then both song sets are the same
    for song in songs1:
        pieces1 = sets1[song]['pieces']
        pieces2 = sets2[song]['pieces']
        comparative_conditionals = [
            (len(pieces1), len(pieces2), 'COUNT TEST'),
            (pieces1.intersection(pieces2), songs1, 'EQUALITY TEST 1'),
            (pieces1.intersection(pieces2), songs2, 'EQUALITY TEST 2'),
            (pieces1.intersection(pieces2), pieces2.intersection(pieces1), 'EQUALITY TEST 2'),
            (pieces1, pieces2, 'EQUALITY TEST 2'),
        ]
        results = equal_comparator(comparative_conditionals, False)
        [print('warning: {}').format(data) if not data else None for data in results]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--path1', help='Parent directory path where source label files are stored', )
    parser.add_argument('--path2', help='Directory path where label files are going to be stored', )
    parser.add_argument('--path1_label_prefix', help='file name of feature label file', default='labels')
    parser.add_argument('--path2_label_prefix', help='file name of feature label file', default='labels')
    # arg parsing
    args = parser.parse_args()
    path1 = Path(args.path1)
    path2 = Path(args.path2)
    path1_label_prefix = Path(args.path1_label_prefix)
    path2_label_prefix = Path(args.path2_label_prefix)

    paths = [{'path': path1, 'label_prefix': path1_label_prefix},
             {'path': path2, 'label_prefix': path2_label_prefix}]
    print('info: checking labels consistency...')

    # real logic
    # open source already splitted label files according to the format: <label_prefix>.<n_classes>.<set_name>.csv
    set_names = ['train', 'test', 'val']
    set_config = {'train': {'song_ratio': 1.0},
                  'test': {'song_ratio': 0.87},
                  'val': {'song_ratio': 1.0}
                  }

    for set_name in set_names:
        # load source metadata from csv
        path1_filename = '{}.{}.{}.csv'.format(path1_label_prefix, NUMBER_OF_CLASSES, set_name)
        path2_filename = '{}.{}.{}.csv'.format(path2_label_prefix, NUMBER_OF_CLASSES, set_name)
        filenames1, labels1 = load_csv(path1, path1_filename)
        filenames2, labels2 = load_csv(path2, path2_filename)
        print('info: If a test fail, probably this program will crash in further tests')
        print('info: comparing label extracted filenames')
        compare_filenames(filenames1, filenames2)

        ls1 = os.listdir(path1)
        ls2 = os.listdir(path2)
        filenames1 = [filename if 'mp3' in filename or 'npy' in filename else None for filename in ls1]
        filenames2 = [filename if 'mp3' in filename or 'npy' in filename else None for filename in ls2]
        print('info: comparing list dir extracted filenames')
        compare_filenames(filenames1, filenames2)
        #
        # print('info: Song pieces test')
        # # load dest metadata from csv
        # label_filename = '{}.csv'.format(dest_label_prefix)
        # filenames, labels = load_csv(dest_path, label_filename)
        # # shuffle data elements
        # indices = np.arange(len(filenames))
        # np.random.seed(RANDOM_SEED)
        # np.random.shuffle(indices)
        # filenames = np.asarray(filenames[indices])
        # labels = np.asarray(labels[indices])
        # # filter only the songs in the source metadata
        # out_indices = []
        # for data_idx, filename in enumerate(filenames):
        #     label = labels[data_idx]
        #     song_name = filename.split('.')[0]
        #     if song_name in songs:
        #         # if song name is in selected song set, append the index for further filtering
        #         out_indices.append(data_idx)
        # # export filtered filenames and labels to CSV
        # label_filename = '{}.{}.{}.csv'.format(dest_label_prefix, NUMBER_OF_CLASSES, set_name)
        # pd.DataFrame(
        #     {
        #         'filename': filenames[out_indices],
        #         'label': labels[out_indices],
        #     }
        # ).to_csv(dest_path / label_filename)
