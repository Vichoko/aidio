"""
Some audio files in the dataset are too big to be processed in one step,
so they need to be splitted beforehand.
"""

import argparse
# empirically defined 20 MB .mp3 file is big enough to crash feature extractors
import os
from _json import make_encoder
from pathlib import Path
from shutil import copyfile

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from config import RAW_DATA_PATH, AVAIL_MEDIA_TYPES, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--src_path', help='Source path where audio data files are stored', default=Path('..') / RAW_DATA_PATH)
    parser.add_argument('--dest_path', help='Source path where audio data files are stored', default=Path('..') / RAW_DATA_PATH / 'skipped')
    parser.add_argument('--label_file', help='file name of label file', default='labels.csv')
    args = parser.parse_args()
    src_path = Path(args.src_path)
    dest_path = Path(args.dest_path)
    makedirs(dest_path)
    label_filename = args.label_file

    df = pd.read_csv(src_path / label_filename)
    new_filenames = []
    new_labels = []

    df_filenames = set(df['filename'])
    os_filenames = set(os.listdir(src_path))
    # keep the filenames that are in the folder and not in the label file
    dif_filenames = os_filenames - df_filenames

    for filename in dif_filenames:
        ext = filename.split('.')[-1]
        if ext not in AVAIL_MEDIA_TYPES:
            print('info: skipping {}. Extension not recognized.')
            continue

        os.replace(src_path / filename, dest_path / filename)
        print('info: moving {} from {} to {}.'.format(filename, src_path, dest_path))

