# aidio wrapper


# Singing Voice Separation
python aidio.py features --features_path E:\aidio_data\features --feature svs_openunmix --raw_path E:\parsed_singers.v2

# Frame Selection
python aidio.py features --features_path E:\aidio_data\features --feature frame_selection --raw_path E:\parsed_singers.v2




# clean dataset

python commands\clean_dataset.py --src_path E:\aidio_data\features\svs_openunmix --dest_path E:\aidio_data\features\svs_ou_skipped --label_file labels.svs_openunmix.csv

# optimize dataset

python commands\optimize_dataset.py --raw_path E:\parsed_singers.v2