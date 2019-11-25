"""
Some audio files in the dataset are too big to be processed in one step,
so they need to be splitted beforehand.
"""

import argparse
# empirically defined 20 MB .mp3 file is big enough to crash feature extractors
import os
from pathlib import Path
from shutil import copyfile

import librosa
import numpy as np
import pandas as pd

from config import RAW_DATA_PATH, AVAIL_MEDIA_TYPES
from features import FeatureExtractor

size_limit = 20 * 1000000  # in bytes
if 'commands' in os.getcwd():
    os.chdir("..")

def split_song(original_file_name, folder_path, sz_limit):
    """
    Load a song, split in halves and export as independent files.
    :param original_file_name: string
    :param folder_path: pathlib.Path
    :return:
    """
    wav, sr = librosa.load(str(folder_path / original_file_name), sr=None)
    ext = original_file_name.split('.')[-1]

    # wav is np.ndarray [shape=(n,) or (2, n)]
    if len(wav.shape) == 1:
        # if is mono, add dummy dim
        wav = np.expand_dims(wav, axis=0)

    size = wav.shape[1]
    ctr = 0
    while size >= sz_limit:
        # count how many half split (binary) should i do to have enough small parts
        size = size // 2
        ctr += 1

    if ctr == 0:
        return [original_file_name]

    parts = 2 * ctr
    part_size = wav.shape[1] // parts
    file_names = []

    for part_idx in range(parts):
        wav_part = wav[:, part_idx * part_size: (part_idx + 1) * part_size]
        n_file_name = original_file_name.replace(ext, '{}.{}'.format(part_idx, ext))
        n_file_name = FeatureExtractor.save_mp3(wav_part.swapaxes(0, 1), sr, None, folder_path, None, None, None, n_file_name)
        file_names.append(n_file_name)

    return file_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('--raw_path', help='Source path where audio data files are stored', default=RAW_DATA_PATH)
    args = parser.parse_args()
    raw_path = Path(args.raw_path)

    df = pd.read_csv(raw_path / 'labels.csv')


    new_filenames = []
    new_labels = []

    for idx, file_name in enumerate(df['filename']):
        label = df['label'][idx]
        file_type = file_name.split('.')[-1]
        if file_type not in AVAIL_MEDIA_TYPES:
            print('warning: Media type {} was ignored while parsing ({}).'.format(file_type, file_name))
            continue
        try:
            size = os.stat(raw_path / file_name).st_size
            if size >= size_limit:
                print('splitting {}'.format(file_name))
                file_names = split_song(file_name, raw_path, size_limit)
                for s_fn in file_names:
                    new_filenames.append(s_fn)
                    new_labels.append(label)
            else:
                new_filenames.append(file_name)
                new_labels.append(label)
        except FileNotFoundError:
            continue

    # export the labels
    metadata = np.asarray([new_filenames, new_labels]).swapaxes(0, 1)
    df = pd.DataFrame(metadata)
    df.columns = ['filename', 'label']
    # ackup prevoious labels
    copyfile(raw_path / 'labels.csv', raw_path / 'labels.csv.bak')
    df.to_csv(raw_path / 'labels.csv', index=False)
    print('info: new label file has been exported successfully.')

parsed_dataset_path = Path('data')
