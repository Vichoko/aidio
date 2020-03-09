# aidio wrapper


# Singing Voice Separation
python aidio.py features --features_path E:\aidio_data\features --feature svs_openunmix --raw_path E:\parsed_singers.v2

# Frame Selection
python aidio.py features --features_path E:\aidio_data\features --feature frame_selection --raw_path E:\parsed_singers.v2


# Train GMM
python aidio.py model --model gmm --experiment svs_2 --data_path /home/voyanedel/data/data/2d/svs-bin-full --label_filename labels.mfcc.csv

# Train Wavenet

python aidio.py model --model wavenet --experiment svs_5 --data_path /home/voyanedel/data/data/1d/svs-bin-full --label_filename labels.mfcc.csv

# clean dataset
python commands\clean_dataset.py --src_path E:\aidio_data\features\svs_openunmix --dest_path E:\aidio_data\features\svs_ou_skipped --label_file labels.svs_openunmix.csv

# optimize dataset

python commands\optimize_dataset.py --raw_path E:\parsed_singers.v2