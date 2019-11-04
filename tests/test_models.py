import unittest

from numpy.testing import assert_allclose

from features import MelSpectralCoefficientsFeatureExtractor
from loaders import DataManager, ResnetDataManager, TorchVisionDataManager
from models import ResNetV2, Simple2dConvNet
from tests.config import TEST_RAW_DATA_PATH, TEST_FEATURES_DATA_PATH, TEST_MODELS_DATA_PATH

import numpy as np
from keras.models import load_model


class TestResnetModel(unittest.TestCase):

    def setUp(cls):
        super().setUp()
        cls.extractor = MelSpectralCoefficientsFeatureExtractor.from_label_file(
            TEST_RAW_DATA_PATH / 'labels.csv',
            out_path=TEST_FEATURES_DATA_PATH,
            source_path=TEST_RAW_DATA_PATH
        )
        extractor = cls.extractor
        cls.assertEqual(len(extractor.new_labels), 0)
        extractor.transform()
        cls.assertGreater(len(extractor.new_labels), 0)
        extracted_data = np.asarray(extractor.new_labels)
        cls.feature_filenames = extracted_data[:, 0]
        cls.feature_labels = extracted_data[:, 1]

        cls.train_dm, cls.test_dm, cls.dev_dm = ResnetDataManager.init_n_split(
            cls.extractor.feature_name,
            feature_data_path=TEST_FEATURES_DATA_PATH,
            shuffle=True,
            ratio=(0.5, 0.5, 0),
            random_state=42
        )
        cls.test_dm.data_loader()
        cls.model = None

    def tearDown(self):
        super().tearDown()
        self.extractor.remove_feature_files()
        self.train_dm.clean_cache()
        self.test_dm.clean_cache()
        self.dev_dm.clean_cache()
        if self.model is not None:
            self.model.remove_checkpoint()

    def test_init(self):
        x_train = self.train_dm.X
        y_train = self.train_dm.Y
        x_test = self.test_dm.X
        y_test = self.test_dm.Y

        # padding to meet requirement
        self.model = ResNetV2(
            'test_init',
            num_classes=y_train.shape[1],
            input_shape=x_train.shape,
            model_path=TEST_MODELS_DATA_PATH,
            epochs=1
        )

        self.model.train_now(x_train, y_train, x_test, y_test)
        self.model.evaluate(x_test, y_test)

    def test_load_predict(self):
        x_train = self.train_dm.X
        y_train = self.train_dm.Y
        x_test = self.test_dm.X
        y_test = self.test_dm.Y

        self.model = ResNetV2(
            'test_init',
            num_classes=y_train.shape[1],
            input_shape=x_train.shape,
            model_path=TEST_MODELS_DATA_PATH,
            epochs=1
        )

        self.model.train_now(x_train, y_train, x_test, y_test)
        self.model.evaluate(x_test, y_test)
        checkpoint_files, checkpoint_epochs = self.model.checkpoint_files
        self.assertEqual(len(checkpoint_files), len(checkpoint_epochs))
        self.assertEqual(len(checkpoint_files), 1)

        loaded_model = load_model(str(checkpoint_files[0]))
        assert_allclose(
            self.model.forward(x_test),
            loaded_model.predict(x_test),
            1e-5
        )


class TestSimpleConvModel(unittest.TestCase):

    def setUp(cls):
        super().setUp()
        cls.extractor = MelSpectralCoefficientsFeatureExtractor.from_label_file(
            TEST_RAW_DATA_PATH / 'labels.csv',
            out_path=TEST_FEATURES_DATA_PATH,
            source_path=TEST_RAW_DATA_PATH
        )
        extractor = cls.extractor
        cls.assertEqual(len(extractor.new_labels), 0)
        extractor.transform()
        cls.assertGreater(len(extractor.new_labels), 0)
        extracted_data = np.asarray(extractor.new_labels)
        cls.feature_filenames = extracted_data[:, 0]
        cls.feature_labels = extracted_data[:, 1]

        cls.train_dm, cls.test_dm, cls.dev_dm = TorchVisionDataManager.init_n_split(
            cls.extractor.feature_name,
            feature_data_path=TEST_FEATURES_DATA_PATH,
            shuffle=True,
            ratio=(0.5, 0.5, 0),
            random_state=42
        )
        cls.test_dm.data_loader()
        cls.model = None

    def tearDown(self):
        super().tearDown()
        self.extractor.remove_feature_files()
        self.train_dm.clean_cache()
        self.test_dm.clean_cache()
        self.dev_dm.clean_cache()
        # if self.model is not None:
        #     self.model.remove_checkpoint()

    def test_init(self):
        x_train = self.train_dm.X
        y_train = self.train_dm.Y
        x_test = self.test_dm.X
        y_test = self.test_dm.Y

        # padding to meet requirement
        self.model = Simple2dConvNet(
            'test_init',
            num_classes=len(set([label_id for label_id in y_train])),
            input_shape=x_train.shape,
            model_path=TEST_MODELS_DATA_PATH,
            epochs=2,
            batch_size=2,
            model_type='simple_conv_net'
        )

        self.model.train_now(x_train, y_train, x_test, y_test)

    # def test_load_predict(self):
    #     x_train = self.train_dm.X
    #     y_train = self.train_dm.Y
    #     x_test = self.test_dm.X
    #     y_test = self.test_dm.Y
    #
    #     self.model = ResNetV2(
    #         'test_init',
    #         num_classes=y_train.shape[1],
    #         input_shape=x_train.shape,
    #         model_path=TEST_MODELS_DATA_PATH,
    #         epochs=1
    #     )
    #
    #     self.model.train(x_train, y_train, x_test, y_test)
    #     self.model.evaluate(x_test, y_test)
    #     checkpoint_files, checkpoint_epochs = self.model.checkpoint_files
    #     self.assertEqual(len(checkpoint_files), len(checkpoint_epochs))
    #     self.assertEqual(len(checkpoint_files), 1)
    #
    #     loaded_model = load_model(str(checkpoint_files[0]))
    #     assert_allclose(
    #         self.model.predict(x_test),
    #         loaded_model.predict(x_test),
    #         1e-5
    #     )

