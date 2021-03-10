import argparse
import subprocess

import features
import model_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from a data folder to another')
    parser.add_argument('module',
                        help='mode can be features, models)')
    features.add_cli_args(parser)
    model_manager.add_cli_args(parser)

    args = parser.parse_args()
    module = args.module

    print(args)

    if module == 'features':
        module = 'features.py'
        features_path, raw_path, feature_name = features.parse_cli_args(args)
        cmd = ['python',
               module,
               '--features_path', str(features_path),
               '--raw_path', str(raw_path),
               '--feature', str(feature_name)
               ]
    elif module == 'model':
        module = 'model_manager.py'
        model_name, experiment_name, data_path, models_path, label_filename, gpus, mode = model_manager.parse_cli_args(
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
               '--mode', str(mode),
               ]
    else:
        raise NotImplementedError

    # call as subprocess for failure managing
    # errno = -666
    # while errno != 0:
    print('info: calling module {}'.format(module))
    errno = subprocess.call(cmd)
    print('debug: cmd call ended with errno = {}'.format(errno))
