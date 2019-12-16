"""
A helper is a wrapper that joins a DataSet with a Trainer in a more compact way.
"""
import argparse

import pytorch_lightning as ptl

from config import makedirs, FEATURES_DATA_PATH, MODELS_DATA_PATH
from features import MelSpectralCoefficientsFeatureExtractor, SingingVoiceSeparationOpenUnmixFeatureExtractor
from loaders import CepstrumDataset, WaveformDataset
from trainers import L_ResNext50, L_WavenetTransformerClassifier, L_WavenetLSTMClassifier


class AbstractHelper:
    pass


class ResNext50Helper:
    model_name = 'resnext50'

    def __init__(self, experiment_name, parser, features_path, models_path, ):
        self.features_path = features_path
        self.models_path = models_path
        self.experiment_name = experiment_name

        train_dataset, test_dataset, eval_dataset, number_of_classes = CepstrumDataset.init_sets(
            MelSpectralCoefficientsFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        parser = L_ResNext50.add_model_specific_args(parser, None)
        hyperparams = parser.parse_args()

        self.model = L_ResNext50(
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
        self.trainer = ptl.Trainer(
            gpus=hyperparams.gpus,
            distributed_backend=hyperparams.distributed_backend,
            logger=logger,
            default_save_path=save_dir,
            early_stop_callback=None
        )

    def train(self):
        self.trainer.fit(self.model)


class WavenetTransformerHelper:
    model_name = 'wavenet_transformer'

    def __init__(self, experiment_name, parser, features_path, models_path, ):
        self.features_path = features_path
        self.models_path = models_path
        self.experiment_name = experiment_name

        train_dataset, test_dataset, eval_dataset, number_of_classes = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        parser = L_WavenetTransformerClassifier.add_model_specific_args(parser, None)
        hyperparams = parser.parse_args()

        self.model = L_WavenetTransformerClassifier(
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
        self.trainer = ptl.Trainer(
            gpus=hyperparams.gpus,
            distributed_backend=hyperparams.distributed_backend,
            logger=logger,
            default_save_path=save_dir,
            early_stop_callback=None
        )

    def train(self):
        self.trainer.fit(self.model)


class WavenetLSTMHelper:
    model_name = 'wavenet_lstm'

    def __init__(self, experiment_name, parser, features_path, models_path, ):
        self.features_path = features_path
        self.models_path = models_path
        self.experiment_name = experiment_name

        train_dataset, test_dataset, eval_dataset, number_of_classes = WaveformDataset.init_sets(
            SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
            features_path,
            ratio=(0.5, 0.25, 0.25)
        )

        parser = L_WavenetLSTMClassifier.add_model_specific_args(parser, None)
        hyperparams = parser.parse_args()

        self.model = L_WavenetLSTMClassifier(
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
        self.trainer = ptl.Trainer(
            gpus=hyperparams.gpus,
            distributed_backend=hyperparams.distributed_backend,
            logger=logger,
            default_save_path=save_dir,
            early_stop_callback=None
        )

    def train(self):
        self.trainer.fit(self.model)


class GMMClassifierHelper:
    """
    Provide train and evaluation methods.
    Connect datasets and loaders with models.
    """

    def __init__(self, num_classes, train_dataset, eval_dataset, test_dataset):
        # model
        self.model = GMMClassifier(num_classes=num_classes)
        # data sets
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        self.eval_ds = eval_dataset
        # data loaders
        self.train_dl = DataLoader(
            self.train_ds,
            num_workers=NUM_WORKERS,
            batch_sampler=ClassSampler(num_classes, self.train_ds.labels),
            collate_fn=ClassSampler.collate_fn
        )
        self.test_dl = DataLoader(
            self.test_ds,
            num_workers=NUM_WORKERS,
            batch_sampler=ClassSampler(num_classes, self.test_ds.labels),
            collate_fn=ClassSampler.collate_fn
        )
        self.eval_dl = DataLoader(
            self.eval_ds,
            num_workers=NUM_WORKERS,
            batch_sampler=ClassSampler(num_classes, self.eval_ds.labels),
            collate_fn=ClassSampler.collate_fn
        )


helpers = {
    ResNext50Helper.model_name: ResNext50Helper,
    WavenetTransformerHelper.model_name: WavenetTransformerHelper,
    WavenetLSTMHelper.model_name: WavenetLSTMHelper

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

    # if model == 'waveNetLstm':
    #     train_dataset, test_dataset, eval_dataset, number_of_classes = WaveformDataset.init_sets(
    #         SingingVoiceSeparationOpenUnmixFeatureExtractor.feature_name,
    #         features_path,
    #         ratio=(0.5, 0.25, 0.25)
    #     )
    #
    #     train_dataloader = DataLoader(train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
    #                                   num_workers=NUM_WORKERS)
    #     test_dataloader = DataLoader(test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    #     val_dataloader = DataLoader(eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    #
    #     # model hyper parameters should be modified in config file
    #     input_shape = (WAVENET_BATCH_SIZE, WAVEFORM_NUM_CHANNELS, WAVEFORM_MAX_SEQUENCE_LENGTH)
    #     model = WaveNetBiLSTMClassifier(
    #         experiment_name,
    #         num_classes=number_of_classes,
    #         input_shape=input_shape,
    #         device_name=device_name
    #     )
    #     model = model.load_checkpoint()
    #     model.train_now(train_dataset, eval_dataset)
    #     model.evaluate(test_dataset)
