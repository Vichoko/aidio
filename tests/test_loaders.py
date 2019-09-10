import unittest

import numpy as np

from features import MelSpectralCoefficientsFeatureExtractor
from loaders import DataManager, ResnetDataManager
from tests.config import TEST_RAW_DATA_PATH, TEST_FEATURES_DATA_PATH


class TestDataManager(unittest.TestCase):

    def setUp(cls, dm_cls=DataManager):
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
        cls.feature_filenames = extracted_data[:, 0]
        cls.feature_labels = extracted_data[:, 1]
        cls.data_type = 'manual'
        cls.dm = dm_cls(extractor.feature_name, cls.data_type, feature_data_path=TEST_FEATURES_DATA_PATH)

    def tearDown(self):
        super().tearDown()
        self.extractor.remove_feature_files()
        self.dm.clean_cache()

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

        self.assertEqual(dm.data_loader, None)
        dm.load_all(lazy=True, cache=False)
        self.assertNotEqual(dm.data_loader, None)

        # here, no side-effects should be felt
        self.assertEqual(dm.X, None)
        self.assertEqual(dm.Y, None)

        # now we evaluate
        if not_implemented:
            with self.assertRaises(NotImplementedError):
                dm.data_loader()
        else:
            dm.data_loader()

        # side-effects should be noticeable
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

    def test_init_n_split(self):
        with self.assertRaises(NotImplementedError):
            train_dm, test_dm, dev_dm = self.dm.init_n_split(
                self.extractor.feature_name,
                feature_data_path=TEST_FEATURES_DATA_PATH,
                shuffle=True,
                ratio=(0.5, 0.3, 0.2),
                random_state=42
            )
        with self.assertRaises(NotImplementedError):
            train_dm, test_dm, dev_dm = self.dm.init_n_split(
                self.extractor.feature_name,
                feature_data_path=TEST_FEATURES_DATA_PATH,
                shuffle=True,
                ratio=(0.5, 0.5, 0),
                random_state=42
            )
        with self.assertRaises(NotImplementedError):
            train_dm, test_dm, dev_dm = self.dm.init_n_split(
                self.extractor.feature_name,
                feature_data_path=TEST_FEATURES_DATA_PATH,
                shuffle=True,
                ratio=(0.5, 0, 0.5),
                random_state=42
            )

        print('end')


class TestResnetDataManager(TestDataManager):

    def setUp(cls, dm_cls=ResnetDataManager):
        # super instances data
        super().setUp(dm_cls)

    def tearDown(self):
        super().tearDown()

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

    def test_init_n_split(self):
        def __test_lazyness(self, dev_dm, test_dm, train_dm):
            # train data is filled instantly
            self.assertTrue(train_dm.X is not None)
            self.assertTrue(train_dm.Y is not None)
            # test and dev are lazy
            self.assertTrue(test_dm.X is None)
            self.assertTrue(test_dm.Y is None)
            self.assertTrue(dev_dm.X is None)
            self.assertTrue(dev_dm.Y is None)
            self.assertFalse(isinstance(test_dm.X, np.ndarray))
            self.assertFalse(isinstance(test_dm.Y, np.ndarray))
            self.assertFalse(isinstance(dev_dm.X, np.ndarray))
            self.assertFalse(isinstance(dev_dm.Y, np.ndarray))
            # once they are evaluated, they get values
            test_dm.data_loader()
            dev_dm.data_loader()
            # now data should be filled
            self.assertTrue(test_dm.X is not None)
            self.assertTrue(test_dm.Y is not None)
            self.assertTrue(dev_dm.X is not None)
            self.assertTrue(dev_dm.Y is not None)
            self.assertIsInstance(test_dm.X, np.ndarray)
            self.assertIsInstance(test_dm.Y, np.ndarray)
            self.assertIsInstance(dev_dm.X, np.ndarray)
            self.assertIsInstance(dev_dm.Y, np.ndarray)
            # data should match all data
            self.assertEqual(len(self.feature_filenames), len(self.feature_labels))
            self.assertEqual(len(self.feature_filenames), len(train_dm.X) + len(test_dm.X) + len(dev_dm.X))

        train_dm, test_dm, dev_dm = self.dm.init_n_split(
            self.extractor.feature_name,
            feature_data_path=TEST_FEATURES_DATA_PATH,
            shuffle=True,
            ratio=(0.5, 0.3, 0.2),
            random_state=42
        )
        __test_lazyness(self, dev_dm, test_dm, train_dm)
        train_dm.clean_cache()
        test_dm.clean_cache()
        dev_dm.clean_cache()

        train_dm, test_dm, dev_dm = self.dm.init_n_split(
            self.extractor.feature_name,
            feature_data_path=TEST_FEATURES_DATA_PATH,
            shuffle=True,
            ratio=(0.5, 0.5, 0),
            random_state=42
        )
        __test_lazyness(self, dev_dm, test_dm, train_dm)
        self.assertTrue(dev_dm.X is not None)
        train_dm.clean_cache()
        test_dm.clean_cache()
        dev_dm.clean_cache()

        train_dm, test_dm, dev_dm = self.dm.init_n_split(
            self.extractor.feature_name,
            feature_data_path=TEST_FEATURES_DATA_PATH,
            shuffle=True,
            ratio=(0.5, 0, 0.5),
            random_state=42
        )
        __test_lazyness(self, dev_dm, test_dm, train_dm)
        train_dm.clean_cache()
        test_dm.clean_cache()
        dev_dm.clean_cache()
