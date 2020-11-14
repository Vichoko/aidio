"""
Call on a directory that contain y_pred and y_target npy tensors.
Calculates many classification metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import sklearn



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--src_path', help='', )
    args = parser.parse_args()
    src_path = Path(args.src_path)

    y_target = np.load(str(src_path / 'y_target.npy'))  # truth values (n_data)
    y_pred = np.load(str(src_path / 'y_pred.npy'))  # predicted values (n_data, n_classes)
    y_pred_nominal = np.argmax(y_pred, axis=1) # predicted values (n_data)






if __name__ == '__main__':
    main()
