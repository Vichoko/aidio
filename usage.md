# aidio wrapper
## Singing Voice Separation
```
python aidio.py features --features_path E:\aidio_data\features --feature svs_openunmix --raw_path E:\parsed_singers.v2
```

## Frame Selection
```
python aidio.py features --features_path E:\aidio_data\features --feature frame_selection --raw_path E:\parsed_singers.v2
```

# Train GMM
```
python aidio.py model --model gmm --experiment svs_2 --data_path /home/voyanedel/data/data/2d/svs-bin-full --label_filename labels.mfcc.csv
```

# Train Wavenet
## Wavenet Vanilla
```
python aidio.py model --model wavenet --data_path /home/vichoko/data/data/1d/svs-bin-full --label_filename labels.csv --gpus [0] --experiment foo
```

## Wavenet Transformer
```
python aidio.py model --model wavenet --data_path /home/vichoko/data/data/1d/svs-bin-full --label_filename labels.csv --gpus [0,1] --experiment foo
```

## Wavenet BiLSTM
```
python aidio.py model --model wavenet_lstm --data_path /home/vichoko/data/data/1d/svs-bin-full --label_filename labels.csv --gpus [0] --experiment foo
```

## GMM
```
python aidio.py model --model gmm --data_path /home/vichoko/data/data/2d/svs-bin-full --label_filename labels.csv --gpus [0] --experiment foo
```

# clean dataset
```
python commands\clean_dataset.py --src_path E:\aidio_data\features\svs_openunmix --dest_path E:\aidio_data\features\svs_ou_skipped --label_file labels.svs_openunmix.csv
```

# optimize dataset
```
python commands\optimize_dataset.py --raw_path E:\parsed_singers.v2
```

# Known Errors

If you get
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with %s library."
                            "\n\tTry to import numpy first or set the threading layer accordingly.
```
Then export this variable
```
export MKL_THREADING_LAYER=GNU
```

# 1D to 2D
Note: This is method is intended to import wav files from numpy.load method. So file names must be in .npy format.

1. Set the correct ```NUMBER_OF_CLASSES``` on ```config.py``` to match desired label file to be exactly exported.
2. Run ```wav_to_mfcc.py``` and wait to transform each:

```
python commands/wav_to_mfcc.py --src_path /home/vichoko/data/data/1d/svs-svd-bin-full --dest_path /home/vichoko/data/data/2d/svs-svd-N --src_label_prefix labels
```

# Subset Data Folder

1. Set the correct ```NUMBER_OF_CLASSES``` on ```config.py``` to match desired label file to be exactly exported.
2. Run ```copy_data_by_label.py``` and wait to transform each:

```
python commands/copy_data_by_label.py --src_path /home/vichoko/data/data/1d/svs-svd-bin-full --dest_path /home/vichoko/data/data/1d/svs-svd-N --src_label_prefix labels
```


# sync data folders through SSH
```
rsync -a /opt/file.zip user@12.12.12.12:/var/www/
```

rsync -avz -e "ssh -p $portNumber" user@remoteip:/path/to/files/ /local/path/