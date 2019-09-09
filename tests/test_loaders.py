import unittest

import numpy as np

from features import MelSpectralCoefficientsFeatureExtractor
from loaders import DataManager, ResnetDataManager
from tests.config import TEST_RAW_DATA_PATH, TEST_FEATURES_DATA_PATH


class TestDataManager(unittest.TestCase):

    def setUp(cls):
        super().setUp()
        cls.extractor = MelSpectralCoefficientsFeatureExtractor.from_label_file(
            TEST_RAW_DATA_PATH / 'labels.csv',
            out_path=TEST_FEATURES_DATA_PATH,
            raw_path=TEST_RAW_DATA_PATH
        )
        extractor = cls.extractor
        cls.assertEqual(len(extractor.new_labels), 0)
        extractor.transform()
        cls.assertGreater(len(extractor.new_labels), 0)
        extracted_data = np.asarray(extractor.new_labels)
        feature_filenames = extracted_data[:, 0]
        cls.feature_labels = extracted_data[:, 1]
        cls.data_type = 'manual'
        cls.dm = DataManager(extractor.feature_name, cls.data_type, feature_data_path=TEST_FEATURES_DATA_PATH)

    def test_init(self):
        dm = self.dm
        extractor = self.extractor

        self.assertEqual(dm.X, None)
        self.assertEqual(dm.Y, None)
        self.assertEqual(dm.data_loader, None)
        self.assertEqual(dm.data_type, self.data_type)
        self.assertEqual(dm.feature_data_path, TEST_FEATURES_DATA_PATH / extractor.feature_name)

    def test_load_all(self, not_implemented=True):
        dm = self.dm
        extractor = self.extractor
        if not_implemented:
            with self.assertRaises(NotImplementedError):
                dm.load_all(cache=False)
        else:
            dm.load_all(cache=False)
        self.assertIsInstance(dm.X, np.ndarray)
        self.assertIsInstance(dm.Y, np.ndarray)

        self.assertEqual(dm.sample_num, dm.X.shape[0])
        self.assertEqual(dm.sample_num, len(extractor.new_labels))
        self.assertEqual(dm.data_type, self.data_type)
        self.assertEqual(dm.feature_data_path, TEST_FEATURES_DATA_PATH / extractor.feature_name)

    def test_lazy_load_all(self, not_implemented=True):
        dm = self.dm
        extractor = self.extractor
        dm.load_all(lazy=True, cache=False)
        if not_implemented:
            with self.assertRaises(NotImplementedError):
                dm.data_loader()
        else:
            dm.data_loader()

        self.assertIsInstance(dm.X, np.ndarray)
        self.assertIsInstance(dm.Y, np.ndarray)

        self.assertEqual(dm.sample_num, dm.X.shape[0])
        self.assertEqual(dm.sample_num, len(extractor.new_labels))
        self.assertEqual(dm.data_type, self.data_type)
        self.assertEqual(dm.feature_data_path, TEST_FEATURES_DATA_PATH / extractor.feature_name)

    def test_one_shot(self):
        dm = self.dm
        with self.assertRaises(NotImplementedError):
            dm.load_all(cache=False)

        old_y = dm.Y
        dm.Y_to_one_shot()

        self.assertFalse(np.asarray(old_y == dm.Y).any())
        self.assertIsInstance(dm.Y, np.ndarray)
        self.assertEqual(dm.Y.shape[0], dm.sample_num)
        self.assertEqual(dm.Y.shape[1], len(set(self.feature_labels)))

    def test_batch_iterator(self):
        with self.assertRaises(NotImplementedError):
            self.dm.batch_iterator()

    def test_get_feed_dict(self):
        with self.assertRaises(NotImplementedError):
            self.dm.get_feed_dict(None, self.data_type)


class TestResnetDataManager(TestDataManager):

    def setUp(cls):
        super().setUp()
        cls.dm = ResnetDataManager(cls.extractor.feature_name, cls.data_type, feature_data_path=TEST_FEATURES_DATA_PATH)

    def test_init(self):
        dm = self.dm
        extractor = self.extractor
        dm.load_all(cache=False)

        self.assertIsInstance(dm.X, np.ndarray)
        self.assertIsInstance(dm.Y, np.ndarray)

        self.assertEqual(dm.sample_num, dm.X.shape[0])
        self.assertEqual(dm.sample_num, len(extractor.new_labels))
        self.assertEqual(dm.data_type, self.data_type)
        self.assertEqual(dm.feature_data_path, TEST_FEATURES_DATA_PATH / extractor.feature_name)

    def test_one_shot(self):
        dm = self.dm

        # load_all calls format_all who calls one_shot internally
        dm.load_all(cache=False)

        self.assertIsInstance(dm.Y, np.ndarray)
        self.assertEqual(dm.Y.shape[0], dm.sample_num)
        self.assertEqual(dm.Y.shape[1], len(set(self.feature_labels)))

    def test_load_all(self, not_implemented=True):
        super().test_load_all(False)

    def test_lazy_load_all(self, not_implemented=True):
        super().test_lazy_load_all(False)
