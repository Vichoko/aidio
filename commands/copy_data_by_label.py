"""
Copy the sound and label files fron a dir to antoher based on given labelfile
"""
import argparse
import inspect
# empirically defined 20 MB .mp3 file is big enough to crash feature extractors
import os
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import NUMBER_OF_CLASSES


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
    parser.add_argument('--dest_label_prefix', help='file name of feature label file', default='labels')
    # arg parsing
    args = parser.parse_args()
    src_path = Path(args.src_path)
    dest_path = Path(args.dest_path)
    src_label_prefix = Path(args.src_label_prefix)
    dest_label_prefix = Path(args.dest_label_prefix)

    set_names = ['train', 'test', 'val']
    for set_name in set_names:
        # load source metadata from csv
        # copy label file
        label_filename = '{}.{}.{}.csv'.format(src_label_prefix, NUMBER_OF_CLASSES, set_name)
        copyfile(
            src_path / label_filename,
            dest_path / label_filename
        )
        # copy files
        filenames, labels = load_csv(src_path, label_filename)
        for data_idx, filename in enumerate(filenames):
            copyfile(
                src_path / filename,
                dest_path / filename
            )
