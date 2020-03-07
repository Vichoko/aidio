import argparse
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
from config import SR

MAX_CLASS_NUMBER = 0  # Number of classes; 0 is all possible
SPLIT_AUDIO_LENGTH = 20  # Second


def main():
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
    class_set = set()
    assert len(filenames) == len(labels)
    for idx, filename in enumerate(filenames):
        label = labels[idx]
        # only accept 10 classes-logic ahead
        if MAX_CLASS_NUMBER == 0:
            # if is 0, it means all possible classes
            pass
        elif len(class_set) < 10:
            # if less than 10 classes are seen, pass
            class_set.add(label)
        elif len(class_set) >= 10 and label not in class_set:
            # if set is full and label is not inside, ignore
            continue
        else:
            # if set is full and label is inside
            pass
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

            new_filename = filename.replace(extension, '{}.npy'.format(slice_idx))
            np.save(
                out_path / new_filename,
                new_wav
            )
            print('info: {} exported successfully.'.format(new_filename))
            new_filenames.append(new_filename)
            new_labels.append(label)
    df = pd.DataFrame(np.asarray([new_filenames, new_labels]).swapaxes(0, 1))
    df.columns = ['filename', 'label']
    df.to_csv(out_path / label_filename, index=False)
    print('info: New {} exported successfully.'.format(label_filename))


if __name__ == '__main__':
    main()
