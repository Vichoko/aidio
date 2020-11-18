"""
Call on a directory that contain y_pred and y_target npy tensors.
Calculates many classification metrics.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix

archs = ['wnlstm', 'gmm']
nclasses = [2, 4, 8, 16, 32]
font_size = [7, 7, 7, 7, 7]
fig_size = [(10, 8), (10, 8), (10, 8), (10, 8), (10, 8), ]


def get_metrics(src_path, arch, nclass, font_size, figsize):
    src_path = src_path / arch / str(nclass)
    labels = pd.read_csv(src_path / 'labels.test.csv')

    y_target = np.load(str(src_path / 'y_target.npy'))  # truth values (n_data)
    y_pred = np.load(str(src_path / 'y_pred.npy'))  # predicted values (n_data, n_classes)
    y_pred_nominal = np.argmax(y_pred, axis=1)  # predicted values (n_data)

    label_names = [None] * (max(y_target) + 1)
    for idx, elem in enumerate(y_target):
        label_names[elem] = labels['label'][idx]
        if None not in label_names:
            break

    confmat = confusion_matrix(y_target, y_pred_nominal, normalize='true')
    report = classification_report(y_target, y_pred_nominal, output_dict=True)
    # cm_display = ConfusionMatrixDisplay(confmat).plot()

    df_cm = pd.DataFrame(confmat, index=label_names,
                         columns=label_names)
    df_cm = df_cm.round(3)
    df_cm = df_cm * 100
    plt.figure(figsize=figsize)
    hm = sn.heatmap(df_cm, annot=True, annot_kws={"size": font_size}, cmap="tab20b",
            cbar_kws={'format': '%.0f%%'} )
    plt.show()
    # hm.get_figure().savefig(src_path / 'confmat.png')
    # plt.savefig(src_path / 'confmat.png')
    out_dict = {'confmat': confmat.tolist(), 'report': report, 'label_names': label_names}
    json.dump(out_dict,
              open(src_path / 'metrics.json', 'w'),
              indent=4)
    print('info: metrics exported to json file')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--src_path', help='', default='E:\\results')
    args = parser.parse_args()
    src_path = Path(args.src_path)
    for arch in archs:
        for idx, nclass in enumerate(nclasses):
            get_metrics(src_path, arch, nclass, font_size[idx], fig_size[idx])
    # get_metrics(src_path, 'wnlstm', 32, 6, (8, 6))


if __name__ == '__main__':
    main()
