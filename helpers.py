"""
A helper is a wrapper that joins a DataSet with a Trainer in a more compact way.
"""
import argparse
import json
import pathlib

import numpy as np
import pytorch_lightning as ptl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from config import makedirs, MODELS_DATA_PATH, RAW_DATA_PATH, EARLY_STOP_PATIENCE
from loaders import CepstrumDataset, WaveformDataset, ExperimentDataset, ClassSampler
from torch_models import GMMClassifier
from trainers import L_ResNext50, L_WavenetTransformerClassifier, L_WavenetLSTMClassifier, L_GMMClassifier, \
    L_WavenetClassifier


def mfcc_test(dataset, number_of_classes):
    """

    :param dataset:
    :param number_of_classes:
    :return:
    """
    # gmms = [sklearn.mixture.GaussianMixture(n_components=64) for _ in range(number_of_classes)]
    dataloader = DataLoader(dataset, num_workers=4,
                            batch_sampler=ClassSampler(number_of_classes, dataset.labels),
                            collate_fn=ClassSampler.collate_fn
                            )
    # load data to ram
    data = list(dataloader)
    train_data = [None] * number_of_classes
    number_of_samples = 65
    module = GMMClassifier(number_of_classes)

    for class_data in data:
        class_id = class_data['y']
        class_mfcc = class_data['x']
        train_data[class_id] = class_mfcc[:]
        module.fit(train_data[class_id], class_data['y'])

    accuracy = [[]] * number_of_classes
    for data_batch in dataset:
        x = data_batch['x'].squeeze(0).permute(1, 0)
        label = data_batch['y'].item()
        y = module.forward(x)
        sm = torch.nn.Softmax()
        y = sm.forward(y)
        predicted_class = y.max(0)[1]
        if predicted_class == label:
            # True Positive
            accuracy[label].append(1)
        else:
            # False positive
            accuracy[label].append(0)
    for class_idx, accuracies in enumerate(accuracy):
        print('info: Class {} has {} accuracy'.format(class_idx, np.asarray(accuracies).mean()))


class AbstractHelper:
    model_name = 'UnnamedHelper'
    dataset = ExperimentDataset
    lightning_module = L_ResNext50
    dataset_ratios = (0.5, 0.25, 0.25)

    def __init__(self, experiment_name, parser, data_path, label_filename, models_path, dummy_mode=False):
        """
        The helper manages model interaction with data.
        :param experiment_name: Name of the experiment
        :param parser: Argparse to add further paramters to the cli
        :param data_path: Absolute path where the data is stored
        :param label_filename: File name of the csv file that contains the filename and label data.
        :param models_path: Path to where model data is stored.
        """
        self.models_path = models_path
        self.experiment_name = experiment_name
        train_dataset, test_dataset, eval_dataset, number_of_classes = self.dataset.init_sets(
            data_path,
            label_filename,
            ratio=self.dataset_ratios,
            dummy_mode=dummy_mode
        )
        #
        # if dummy_mode:
        #     mfcc_test(train_dataset, number_of_classes)
        #     return
        parser = self.lightning_module.add_model_specific_args(parser, None)
        hyperparams = parser.parse_args()
        self.module = self.lightning_module(
            hyperparams,
            number_of_classes,
            train_dataset,
            eval_dataset,
            test_dataset,
        )
        gpus = json.loads(hyperparams.gpus)
        save_dir = models_path / model_name / experiment_name
        makedirs(save_dir)
        logger = ptl.logging.TestTubeLogger(
            save_dir=save_dir,
            version=1  # An existing version with a saved checkpoint
        )
        if 'distributed_backend' not in hyperparams:
            hyperparams.distributed_backend = 'dp'

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=EARLY_STOP_PATIENCE,
            verbose=False,
            mode='min'
        )
        self.trainer = ptl.Trainer(
            gpus=gpus if len(gpus) else 0,
            distributed_backend=hyperparams.distributed_backend,
            logger=logger,
            default_save_path=save_dir,
            early_stop_callback=early_stop_callback,
            nb_sanity_val_steps=0,
            amp_level='O2',
            use_amp=False
        )

    def train(self):
        self.trainer.fit(self.module)

    def evaluate(self):
        """
        Run a set of evaluations to obtain metrics of SID task.
        Metrics:
            Accuracy
                Accuracy per class
                Accuracy total
            Recall
                Per Class
                Total
            Error
        :return:
        """

        return


# folder name of music files after some processing (raw, svs, svd, svs+svd, etc)
# raw_feature_name = 'svs'


class WavenetTransformerHelper(AbstractHelper):
    model_name = 'wavenet_transformer'
    dataset = WaveformDataset
    # source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_WavenetTransformerClassifier


class WavenetLSTMHelper(AbstractHelper):
    model_name = 'wavenet_lstm'
    dataset = WaveformDataset
    # source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_WavenetLSTMClassifier


class WavenetHelper(AbstractHelper):
    model_name = 'wavenet'
    dataset = WaveformDataset
    # source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_WavenetClassifier


class GMMClassifierHelper(AbstractHelper):
    model_name = 'gmm'
    dataset = CepstrumDataset
    # source_feature_name = MelSpectralCoefficientsFeatureExtractor.feature_name
    lightning_module = L_GMMClassifier

    def train(self):
        """
        The lighning_module of GMM has special behaviour. So it has trained boolean
        attribute and save_model methods.
        :return:
        """
        if self.module.trained:
            print('warning: Trying to trained an gmm loaded from disc and already trained. ignoring...')
            return
        super().train()
        self.module.trained = True
        self.module.save_model()


class ResNext50Helper(AbstractHelper):
    model_name = 'resnext50'
    dataset = CepstrumDataset
    # source_feature_name = MelSpectralCoefficientsFeatureExtractor.feature_name
    lightning_module = L_ResNext50


helpers = {
    ResNext50Helper.model_name: ResNext50Helper,
    WavenetTransformerHelper.model_name: WavenetTransformerHelper,
    WavenetLSTMHelper.model_name: WavenetLSTMHelper,
    GMMClassifierHelper.model_name: GMMClassifierHelper,
    WavenetHelper.model_name: WavenetHelper
}


def add_cli_args(parser):
    parser.add_argument(
        '--data_path',
        help='Source path where input data files are stored',
        default=RAW_DATA_PATH
    )
    parser.add_argument(
        '--label_filename',
        help='File name of label file (default is labels.csv)',
        default='labels.csv'
    )
    parser.add_argument(
        '--model_path',
        help='Path where the model data is going to be stored.',
        default=MODELS_DATA_PATH
    )
    parser.add_argument(
        '--model',
        help='Model name. (Ej. resnext50, gmm, transformer)',
        default=GMMClassifierHelper.model_name,
        # required=True
    )
    parser.add_argument(
        '--experiment',
        help='experiment identifier',
        default='unnamed_experiment'
    )
    parser.add_argument('--gpus', default='[]', type=str)


def parse_cli_args(args):
    model_name = args.model
    experiment_name = args.experiment
    data_path = pathlib.Path(args.data_path)
    models_path = pathlib.Path(args.model_path)
    label_filename = args.label_filename
    gpus = json.loads(args.gpus)
    return model_name, experiment_name, data_path, models_path, label_filename, gpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model from features from a data folder',
        add_help=False
    )
    add_cli_args(parser)
    args = parser.parse_args()
    model_name, experiment_name, data_path, models_path, label_filename, _ = parse_cli_args(args)

    helper_class = helpers[model_name]
    helper = helper_class(
        experiment_name,
        parser,
        data_path,
        label_filename,
        models_path,
        dummy_mode=False
    )
    helper.train()
    print('helper ended')
