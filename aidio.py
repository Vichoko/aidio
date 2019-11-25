import argparse
import subprocess

from config import RAW_DATA_PATH, FEATURES_DATA_PATH
from features import SingingVoiceSeparationOpenUnmixFeatureExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('mode',
                        help='mode can be features, models)')
    parser.add_argument(
        '--raw_path',
        help='Source path where audio data files are stored',
        default=RAW_DATA_PATH
    )
    parser.add_argument(
        '--features_path',
        help='Output path where exported data will be placed',
        default=FEATURES_DATA_PATH
    )
    # parser.add_argument('--label_filename', help='Source path where label file is stored', default='labels.csv')

    parser.add_argument(
        '--feature',
        help='name of the feature to be extracted (options: mfsc, leglaive)',
        default=SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    )

    args = parser.parse_args()
    mode = args.mode
    features_path = args.features_path
    raw_path = args.raw_path
    feature_name = args.feature
    print(args)

    if mode == 'features':
        module = 'features.py'
    else:
        raise NotImplementedError

    # call as subprocess for failure managing
    errno = False
    while errno != 0:
        print('info: calling module {}'.format(module))
        cmd = ['python',
               module,
               '--features_path', features_path,
               '--raw_path', raw_path,
               '--feature', feature_name
               ]
        errno = subprocess.call(cmd)
