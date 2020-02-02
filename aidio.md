# Usage examples for aidio.py suite

## Help

´´´
> python aidio.py --help
usage: aidio.py [-h] [--raw_path RAW_PATH] [--features_path FEATURES_PATH]
                [--feature FEATURE]
                mode

Extract features from a data folder to another

positional arguments:
  mode                  mode can be features, models)

optional arguments:
  -h, --help            show this help message and exit
  --raw_path RAW_PATH   Source path where audio data files are stored
  --features_path FEATURES_PATH
                        Output path where exported data will be placed
  --feature FEATURE     name of the feature to be extracted (options: mfsc,
                        leglaive)
´´´

## Feature extraction

### MFCC extraction

´´´
python aidio.py features --raw_path /data/svs/ --feature mfcc
´´´

## ML Model Interaction

´´´
python aidio.py model --model wavenet --experiment svs --data_path /home/voyanedel/data/data/svs-bin --gpus [0]
python aidio.py model --model wavenet_transformer --experiment svs --data_path /home/voyanedel/data/data/svs-bin-full --gpus [1]
python aidio.py model --model gmm --experiment svs_1 --data_path /home/voyanedel/data/data/2d/svs --label_filename labels.mfcc.csv
´´´

# Commands
´´´
python aidio.py features --raw_path /data/svs/ --feature mfcc
´´´
