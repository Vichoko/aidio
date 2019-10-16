import argparse
import json
import pathlib
from os import listdir
from os.path import isdir
from shutil import copy2

import numpy as np
import pandas as pd

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


file_counter = 0


def parse_generic_tree(source_dir, dest_dir, move=False):
    """
    Parse a more generic schema that enforces:
    * First folder is singer name
    * Can have any level of nesting inside,
        * Every non-leaf node is a metadata (Example, artist, album, type of album (EPs, Studio, Live, etc)
        * Every leaf node is a file, if accepted format then it's considered.

    Parse folder schema:
    * Singer-name/
        + Metadata1-1/
            - FileA
        + Metadata1-2/
            + Metadata2-1/
                - FileB
            - FileC
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

    def recursive_song_finder(root_dir, metadata_list, singer_name, move=False):
        """
        Iterate over files and directories in any directory, parsing music files and calling itself on
        nested directories
        :param root_dir: pathlib.Path object
        :param metadata_list: List of metadata as Strings
        :param singer_name: Singer name as string
        :return: List of file data as dictionaries (one element per file)
        """
        global file_counter
        parsed_data = []
        for file_name in listdir(root_dir):
            if isdir(root_dir / file_name):
                # is intern node
                # recursive call
                recursive_metadata_list = metadata_list.copy()
                # add folder name as metadata for recursive call
                recursive_metadata_list.append(str(file_name))
                # recursive call and append results to parsed data
                recursive_parsed_data = recursive_song_finder(root_dir / file_name, recursive_metadata_list,
                                                              singer_name)
                parsed_data.extend(recursive_parsed_data)
            else:
                # is leaf
                # check file extension and parse if is audio
                file_type = file_name.split('.')[-1]
                if file_type not in AVAIL_MEDIA_TYPES:
                    print('warning: Media type {} was ignored while parsing ({}).'.format(file_type, file_name))
                    continue
                # create new unique filename
                file_counter += 1
                try:
                    artist = metadata_list[0]
                except IndexError:
                    artist = 'no_artist'
                new_filename = '{}{}_{}.{}'.format(
                    file_counter,
                    singer_name[:10],
                    artist[:10],
                    file_type)
                if move:
                    raise NotImplementedError
                copy2(root_dir / file_name, dest_dir / new_filename)
                print('info: {} successfully exported!'.format(new_filename))
                parsed_data.append({
                    'singer': singer_name,
                    'metadata': metadata_list,
                    'filename': new_filename,
                    'original_filename': file_name
                })
        return parsed_data

    # grab all files and folders in source directory
    singers_folder_names = listdir(source_dir)
    # filter only directories
    singers_folder_names = [singer_name for singer_name in singers_folder_names if isdir(source_dir / singer_name)]

    parsed_data = []
    # iterate over all directoris, i.e. singer folders
    for singer_name in singers_folder_names:
        # parse this singer data recursivelly
        parsed_data.extend(recursive_song_finder(source_dir / singer_name, [], singer_name))

    json.dump(parsed_data, open(dest_dir / 'labels.json', 'w'))
    return 0


def json_to_csv(path_to_json, path_to_csv):
    """
    Convert a label's json to a Feature Extractor compilant CSV.

    The expected CSV format is the following:
    filename, label, more*

    More columns will be discarded.
    :param path_to_json: pathlib.Path or String
    :param path_to_csv: pathlib.Path or String
    :return:
    """
    print('info: transforming json to csv')
    j = json.load(open(path_to_json, 'r', encoding='utf8'))
    df = pd.DataFrame.from_dict(j)
    df = df.rename(columns={'singer': 'label'})
    df.to_csv(open(path_to_csv, 'w', encoding='utf8'), index=False)
    print('info: transformed and exported to {}'.format(path_to_csv))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform a data directory to our standard for further loading.')
    parser.add_argument('source_directory', help='Source path where data is stored', )
    parser.add_argument('out_directory', help='Output path where data is exported', )
    args = parser.parse_args()
    _in = pathlib.Path(args.source_directory)
    _out = pathlib.Path(args.out_directory)
    print('info: from {} to {}'.format(_in, _out))
    parse_generic_tree(_in, _out)
