"""
A helper is a wrapper that joins a DataSet with a Trainer in a more compact way.
"""
import argparse

import pytorch_lightning as ptl

from config import makedirs, FEATURES_DATA_PATH, MODELS_DATA_PATH
from features import MelSpectralCoefficientsFeatureExtractor, SingingVoiceSeparationOpenUnmixFeatureExtractor
from loaders import CepstrumDataset, WaveformDataset
from trainers import L_ResNext50, L_WavenetTransformerClassifier, L_WavenetLSTMClassifier, L_GMMClassifier


class AbstractHelper:
    model_name = 'UnnamedHelper'
    dataset = CepstrumDataset
    source_feature_name = MelSpectralCoefficientsFeatureExtractor.feature_name
    lightning_module = L_ResNext50
    dataset_ratios = (0.5, 0.25, 0.25)

    def __init__(self, experiment_name, parser, features_path, models_path, ):
        self.features_path = features_path
        self.models_path = models_path
        self.experiment_name = experiment_name

        train_dataset, test_dataset, eval_dataset, number_of_classes = self.dataset.init_sets(
            self.source_feature_name,
            features_path,
            ratio=self.dataset_ratios
        )

        parser = self.lightning_module.add_model_specific_args(parser, None)
        hyperparams = parser.parse_args()

        self.module = self.lightning_module(
            hyperparams,
            number_of_classes,
            train_dataset,
            eval_dataset,
            test_dataset
        )
        save_dir = models_path / model_name / experiment_name
        makedirs(save_dir)
        logger = ptl.logging.TestTubeLogger(
            save_dir=save_dir,
            version=1  # An existing version with a saved checkpoint
        )

        if 'gpus' not in hyperparams:
            hyperparams.gpus = 0
        if 'distributed_backend' not in hyperparams:
            hyperparams.distributed_backend = 'dp'

        self.trainer = ptl.Trainer(
            gpus=hyperparams.gpus,
            distributed_backend=hyperparams.distributed_backend,
            logger=logger,
            default_save_path=save_dir,
            early_stop_callback=None
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
raw_feature_name = 'svs'


class WavenetTransformerHelper(AbstractHelper):
    model_name = 'wavenet_transformer'
    dataset = WaveformDataset
    source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_WavenetTransformerClassifier


class WavenetLSTMHelper(AbstractHelper):
    model_name = 'wavenet_lstm'
    dataset = WaveformDataset
    source_feature_name = SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name
    lightning_module = L_WavenetLSTMClassifier


class GMMClassifierHelper(AbstractHelper):
    model_name = 'gmm'
    dataset = CepstrumDataset
    source_feature_name = MelSpectralCoefficientsFeatureExtractor.feature_name
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
    source_feature_name = MelSpectralCoefficientsFeatureExtractor.feature_name
    lightning_module = L_ResNext50


helpers = {
    ResNext50Helper.model_name: ResNext50Helper,
    WavenetTransformerHelper.model_name: WavenetTransformerHelper,
    WavenetLSTMHelper.model_name: WavenetLSTMHelper,
    GMMClassifierHelper.model_name: GMMClassifierHelper,

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model from features from a data folder',
        add_help=False
    )
    parser.add_argument(
        '--model',
        help='Model name. (Ej. resnext50, gmm, transformer)',
        default=WavenetLSTMHelper.model_name,
        # required=True
    )
    parser.add_argument(
        '--experiment',
        help='experiment identifier',
        default='unnamed_experiment'
    )
    args = parser.parse_args()
    model_name = args.model
    experiment_name = args.experiment

    helper_class = helpers[model_name]
    features_path = FEATURES_DATA_PATH
    models_path = MODELS_DATA_PATH
    helper = helper_class(experiment_name, parser, features_path, models_path)
    helper.train()
    print('helper ended')
