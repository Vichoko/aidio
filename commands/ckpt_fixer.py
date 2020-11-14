"""
Remove metric records on state_dict.

"""

import argparse
from pathlib import Path

import torch

state_dict_keys_to_remove = ['test_acc.total', 'test_acc.correct',
                             'val_acc.total', 'val_acc.correct',
                             'train_acc.total', 'val_acc.correct', ]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--src_path', help='', )
    parser.add_argument('--dest_path', help='', )
    args = parser.parse_args()
    src_path = Path(args.src_path)
    dest_path = Path(args.dest_path)
    ckpt = torch.load(src_path)
    print('info: state_dict keys: {}'.format(ckpt['state_dict'].keys()))
    for k in state_dict_keys_to_remove:
        del ckpt['state_dict'][k]
    torch.save(ckpt, dest_path)


if __name__ == '__main__':
    main()
