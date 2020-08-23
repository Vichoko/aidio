"""
A helper is a wrapper that joins a DataSet with a Trainer in a more compact way.
"""
import argparse
import json
import pathlib

import pytorch_lightning as ptl
from pytorch_lightning.loggers import TestTubeLogger

from config import makedirs, MODELS_DATA_PATH, RAW_DATA_PATH
from lightning_modules import L_ResNext50, L_WavenetTransformerClassifier, L_WavenetLSTMClassifier, L_GMMClassifier, \
    L_WavenetClassifier, L_Conv1DClassifier, L_RNNClassifier
from loaders import CepstrumDataset, WaveformDataset, ExperimentDataset


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
        self.save_dir = models_path / model_name / experiment_name
        makedirs(self.save_dir)

        if 'distributed_backend' not in hyperparams:
            hyperparams.distributed_backend = 'dp'

        # todo: connect ddp, fix gpu specification
        # trainer with some optimizations
        self.trainer = ptl.Trainer(
            gpus=gpus if len(gpus) else 0,
            profiler=False,  # for once is good
            auto_scale_batch_size=False,  # i prefer manually
            auto_lr_find=False,  # mostly diverges
            distributed_backend='dp',  # doesnt fill on ddp
            precision=32,  # throws error on 16
            default_root_dir=self.save_dir,
            logger=TestTubeLogger(
                save_dir=self.save_dir,
                version=1  # fixed to one to ensure checkpoint load
            )
        )

    def train(self):
        self.trainer.fit(self.module)

    def test(self):
        self.trainer.test(self.module)

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
        # self.trainer.run_evaluation(test=False)
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


class Conv1dHelper(AbstractHelper):
    model_name = 'conv1d'
    dataset = WaveformDataset
    # source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_Conv1DClassifier


class RNNHelper(AbstractHelper):
    model_name = 'rnn'
    dataset = WaveformDataset
    # source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_RNNClassifier


class GMMClassifierHelper(AbstractHelper):
    model_name = 'gmm'
    dataset = CepstrumDataset
    # source_feature_name = MelSpectralCoefficientsFeatureExtractor.feature_name
    lightning_module = L_GMMClassifier
    dataset_ratios = (.7, .29, .01)

    def __init__(self, experiment_name, parser, data_path, label_filename, models_path, dummy_mode=False):
        super().__init__(experiment_name, parser, data_path, label_filename, models_path, dummy_mode)
        self.module.load_model(self.save_dir)

    def train(self):
        """
        The lighning_module of GMM has special behaviour. So it has trained boolean
        attribute and save_model methods.
        :return:
        """
        if self.module.train_now() == 0:
            self.module.save_model(self.save_dir)
        self.module.eval_now()


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
    WavenetHelper.model_name: WavenetHelper,
    Conv1dHelper.model_name: Conv1dHelper,
    RNNHelper.model_name: RNNHelper,
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
    parser.add_argument(
        '--mode',
        help='can be train, test or dummy (to activate the dummy training).',
        default='train',
        required=False
    )
    parser.add_argument('--gpus', default='[]', type=str)


def parse_cli_args(args):
    model_name = args.model
    experiment_name = args.experiment
    data_path = pathlib.Path(args.data_path)
    models_path = pathlib.Path(args.model_path)
    label_filename = args.label_filename
    gpus = json.loads(args.gpus)
    mode = args.mode
    return model_name, experiment_name, data_path, models_path, label_filename, gpus, mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model from features from a data folder',
        add_help=False
    )
    add_cli_args(parser)
    args = parser.parse_args()
    model_name, experiment_name, data_path, models_path, label_filename, _, mode = parse_cli_args(args)
    helper_class = helpers[model_name]
    helper = helper_class(
        experiment_name,
        parser,
        data_path,
        label_filename,
        models_path,
        dummy_mode=True if mode == 'dummy' else False
    )
    if mode == 'dummy' or mode == 'train':
        helper.train()
    elif mode == 'test':
        helper.test()
    elif mode == 'evaluation':
        helper.evaluate()
    else:
        raise NotImplementedError('model mode not implemented')
    print('helper ended')
