import argparse
import subprocess

import features
import helpers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('mode',
                        help='mode can be features, models)')
    features.add_cli_args(parser)
    helpers.add_cli_args(parser)

    args = parser.parse_args()
    mode = args.mode

    print(args)

    if mode == 'features':
        module = 'features.py'
        features_path, raw_path, feature_name = features.parse_cli_args(args)
        cmd = ['python',
               module,
               '--features_path', str(features_path),
               '--raw_path', str(raw_path),
               '--feature', str(feature_name)
               ]
    elif mode == 'model':
        module = 'helpers.py'
        model_name, experiment_name, data_path, models_path, label_filename, gpus, dummy_mode = helpers.parse_cli_args(
            args)

        cmd = ['python',
               '-W', 'ignore',  # to suppress userwarnings of librosa
               module,
               '--model', str(model_name),
               '--experiment', str(experiment_name),
               '--data_path', str(data_path),
               '--model_path', str(models_path),
               '--label_filename', str(label_filename),
               '--gpus', str(gpus),
               '--dummy_mode' if dummy_mode else '',
               'true' if dummy_mode else '',
               ]
    else:
        raise NotImplementedError

    # call as subprocess for failure managing
    errno = -666
    while errno != 0:
        print('info: calling module {}'.format(module))
        errno = subprocess.call(cmd)
        print('debug: cmd call ended with errno = {}'.format(errno))
