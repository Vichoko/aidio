import pathlib
from os.path import isdir

import pandas as pd
import numpy as np
import argparse

from os import listdir
from shutil import copy2

from config import AVAIL_MEDIA_TYPES


def parse_standard_tree(source_dir, dest_dir, move=False):
    """
    Parse folder schema:
    * Artist1/
        + Album1/
        + Album2/
            - File1
            - File2
    * Artist2

    to:
    + labels.csv
    + File1
    + File2
    ...

    Labels are based on tree information and should be manually checked later.
    :param move:
    :param source_dir: pathlib.Path object
    :param dest_dir: pathlib.Path object
    :return:
    :return:
    """
    labels = []
    song_counter = 0

    artist_tree = listdir(source_dir)
    artist_tree = [element for element in artist_tree if isdir(source_dir / element)]

    for artist_name in artist_tree:
        artist_tree = listdir(source_dir / artist_name)
        aritst_tree = [element for element in artist_tree if isdir(source_dir / artist_name / element)]

        for album_name in aritst_tree:
            album_tree = listdir(source_dir / artist_name / album_name)

            for file_name in album_tree:
                # export and append label
                file_type = file_name.split('.')[-1]
                if file_type not in AVAIL_MEDIA_TYPES:
                    print('warning: Media type {} was ignored while parsing ({}).'.format(file_type, file_name))
                    continue

                song_counter += 1
                out_file_name = '{}_{}_{}.{}'.format(song_counter, artist_name[:4], album_name[:4], file_type).strip()
                labels.append([out_file_name, artist_name, album_name, file_name])
                if move:
                    raise NotImplemented
                copy2(source_dir / artist_name / album_name / file_name, dest_dir / out_file_name)
                print('info: {} successfully exported!'.format(out_file_name))
    df = pd.DataFrame(np.asarray(labels))
    df.columns = ['filename', 'artist', 'album', 'old_name']
    df.to_csv(dest_dir / 'labels.csv', index=False)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform a data directory to our standard for further loading.')
    parser.add_argument('source_directory', help='Source path where data is stored', )
    parser.add_argument('out_directory', help='Output path where data is exported', )
    args = parser.parse_args()
    _in = pathlib.Path(args.source_directory)
    _out = pathlib.Path(args.out_directory)
    print('info: from {} to {}'.format(_in, _out))
    parse_standard_tree(_in, _out)
