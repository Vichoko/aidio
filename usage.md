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
MKL_THREADING_LAYER=GNU
```