import argparse
import pathlib
import pandas as pd


def asisted_singer_label(label_path):
    """
    Assisted parser of labels
    :param label_path: CSV file containing labels exported by parse_dataset.py script
    :return:
    """
    df = pd.read_csv(label_path)
    unique_artists = list(set(df['artist']))
    unique_labels = []
    for artist in unique_artists:
        x = input('¿El artista {} tiene 1 sólo cantante? [S/n]'.format(artist))
        if str.lower(str.strip(x)) == 'n':
            print('Error: El parser no soporta esta feature. Saltando este artista...')
            continue
        y = 1
        x = y - 1
        while x != y:
            x = input('¿Como se llama el cantante de {}?'.format(artist))
            y = input('Reingrese etiqueta para {} (previo: {})'.format(artist, x))
        unique_labels.append(x)
    df['label'] = df.apply(axis=1, func=lambda row: unique_labels[unique_artists.index(row['artist'])])
    df.to_csv(label_path, index=False)
    print('info: updated label file at {}'.format(label_path))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a label file product of parse_dataset scritp.')
    parser.add_argument('label_path', help='Source path where label file is stored', )
    args = parser.parse_args()
    label_file = pathlib.Path(args.label_path)
    print('info: from {}'.format(label_file))
    asisted_singer_label(label_file, )
