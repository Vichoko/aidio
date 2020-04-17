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
    print('dir1 has {} songs.'.format(len(songs1)))
    print('dir2 has {} songs.'.format(len(songs2)))
    print('dir1 has {} filenames.'.format(len(filenames1)))
    print('dir2 has {} filenames.'.format(len(filenames2)))

    def equal_comparator(conditionals, verbose=True):
        flag = True
        for idx, conditional in enumerate(conditionals):
            cond1 = conditional[0]
            cond2 = conditional[1]
            title = conditional[2]
            value = cond1 == cond2
            flag = flag & value
            if verbose:
                print('info: {}'.format(title))
                print('info [{}]: check!'.format(idx)) if value else print(
                    'warning [{}]: FAILED. {} != {}.'.format(idx, cond1, cond2))
        return flag

    print('info: SONG TESTS')
    flag = equal_comparator(comparative_conditionals)
    print('info [SONG TEST]: check!') if flag else print('warning [SONG TEST]: FAILED!')

    # here we assume then both song sets are the same
    print('info: PIECES TESTS')
    flag = True
    for song in songs1:
        pieces1 = sets1[song]['pieces']
        pieces2 = sets2[song]['pieces']
        print('info: song {} has inconsitent pieces'.format(song)) if pieces1 != pieces2 else None
        flag = flag & (pieces1 == pieces2)
    print('info [PIECES TEST]: check!') if flag else print('warning [PIECES TEST]: FAILED!')


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
    print('info: If a test fail, probably this program will crash in further tests')
    for set_name in set_names:
        print('info: TESTS FOR SET = {}'.format(set_name))
        # load source metadata from csv
        path1_filename = '{}.{}.{}.csv'.format(path1_label_prefix, NUMBER_OF_CLASSES, set_name)
        path2_filename = '{}.{}.{}.csv'.format(path2_label_prefix, NUMBER_OF_CLASSES, set_name)
        filenames1, labels1 = load_csv(path1, path1_filename)
        filenames2, labels2 = load_csv(path2, path2_filename)
        print('info: LABEL EXTRACTED FILENAMES')
        compare_filenames(filenames1, filenames2)

    ls1 = os.listdir(path1)
    ls2 = os.listdir(path2)
    filenames1 = [filename if 'mp3' in filename or 'npy' in filename else None for filename in ls1]
    filenames2 = [filename if 'mp3' in filename or 'npy' in filename else None for filename in ls2]
    print('info: LS EXTRACTED FILENAMES')
    compare_filenames(filenames1, filenames2)
