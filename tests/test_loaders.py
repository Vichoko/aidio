import unittest
from math import ceil

import numpy as np

from features import MelSpectralCoefficientsFeatureExtractor
from loaders import DataManager, ResnetDataManager, ADiSANDataManager, TorchVisionDataManager
from tests.config import TEST_RAW_DATA_PATH, TEST_FEATURES_DATA_PATH


class TestDataManager(unittest.TestCase):

    def setUp(cls, dm_cls=DataManager, extractor_cls=MelSpectralCoefficientsFeatureExtractor):
        super().setUp()
        cls.extractor = extractor_cls.from_label_file(
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
        cls.data_type = 'manual'
        cls.dm = dm_cls(extractor.feature_name, cls.data_type, feature_data_path=TEST_FEATURES_DATA_PATH, epochs=1, batch_size=1)

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

    def setUp(cls, dm_cls=ResnetDataManager, extractor_cls=MelSpectralCoefficientsFeatureExtractor):
        # super instances data
        super().setUp(dm_cls, extractor_cls)

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


class TestADiSANDataManager(TestDataManager):

    def setUp(cls, dm_cls=ADiSANDataManager, extractor_cls=MelSpectralCoefficientsFeatureExtractor):
        # super instances data
        super().setUp(dm_cls, extractor_cls)

    def tearDown(self):
        super().tearDown()
        self.dm.clean_cache()

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
        self.assertEqual(len(dm.Y.shape), 1)
        self.assertEqual(dm.Y.shape[0], dm.sample_num)
        self.assertTrue((len(set(self.dm.Y)) > dm.Y).all())

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

    def test_format(self):
        self.dm.load_all()

        self.assertIsInstance(self.dm.X, np.ndarray)
        self.assertIsInstance(self.dm.Y, np.ndarray)

        n_data = len(self.extractor.x)
        sequence_len = 128
        feature_dim = 16
        n_classes = 2

        # X
        self.assertTrue(self.dm.X.shape[0] == n_data)
        self.assertTrue(self.dm.X.shape[1] == sequence_len)
        self.assertTrue(self.dm.X.shape[2] == feature_dim)

        # Y
        self.assertTrue(self.dm.Y.shape[0] == n_data)
        self.assertIsInstance(self.dm.Y[0], np.int32)
        self.assertEqual(len(set(self.dm.Y)), n_classes)

    def test_batch_iterator(self):
        self.dm.load_all()

        self.dm.epochs = 3
        n_data = len(self.dm.X)
        self.assertEqual(n_data, 8)

        for batch_size in range(n_data):
            # test different batch_sizes
            batch_size += 1
            # set test dynamic variables
            step_idx = 0
            self.dm.batch_size = batch_size
            n_batches = ceil(n_data / self.dm.batch_size)

            for batch_data, total_batch_count, epoch_idx, batch_idx in self.dm.batch_iterator():
                x = batch_data['x']
                y = batch_data['y']

                self.assertEqual(total_batch_count, n_batches)

                self.assertEqual(batch_idx, step_idx % n_batches)
                self.assertEqual(epoch_idx, step_idx // n_batches)

                self.assertEqual(len(x), self.dm.batch_size)
                self.assertEqual(len(y), self.dm.batch_size)

                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                self.assertEqual(len(x.shape), 3)
                self.assertEqual(len(y.shape), 1)

                step_idx += 1

    def test_batch_iterator_max_steps(self):
        self.dm.load_all()

        self.dm.epochs = 3
        n_data = len(self.dm.X)
        self.assertEqual(n_data, 8)
        batch_size = 3

        self.dm.batch_size = batch_size
        n_batches = ceil(n_data / self.dm.batch_size)

        max_steps = self.dm.epochs * n_batches
        for idx in range(max_steps):
            idx += 1
            last_step = int(max_steps / idx)
            data_list = list(self.dm.batch_iterator(last_step))
            self.assertEqual(len(data_list), last_step)

    def test_get_feed_dict(self):
        class DummyModel:
            def __init__(self):
                self.batch_embedding_sequence = 'a'
                self.batch_output_labels = 'b'
                self.batch_access_mask = 'c'
                self.is_train = 'd'

        self.dm.load_all()

        for batch_data, total_batch_count, epoch_idx, batch_idx in self.dm.batch_iterator():
            dummy_model = DummyModel()
            dic = self.dm.get_feed_dict(dummy_model, batch_data, 'train')

            self.assertIn(dummy_model.batch_embedding_sequence, dic.keys())
            self.assertIn(dummy_model.batch_output_labels, dic.keys())
            self.assertIn(dummy_model.batch_access_mask, dic.keys())
            self.assertIn(dummy_model.is_train, dic.keys())

            x = dic[dummy_model.batch_embedding_sequence]
            y = dic[dummy_model.batch_output_labels]
            mask = dic[dummy_model.batch_access_mask]
            is_train = dic[dummy_model.is_train]

            self.assertIsInstance(x, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(mask, np.ndarray)
            self.assertIsInstance(is_train, bool)

            self.assertEqual(len(x.shape), 3)
            self.assertEqual(len(mask.shape), 2)
            self.assertEqual(len(y.shape), 1)

            self.assertEqual(x.shape[0], y.shape[0])
            self.assertEqual(x.shape[0], mask.shape[0])
            self.assertEqual(x.shape[1], mask.shape[1])

            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.int32)
            self.assertEqual(mask.dtype, bool)


class TestTorchVisionDataManager(TestDataManager):

    def setUp(cls, dm_cls=TorchVisionDataManager, extractor_cls=MelSpectralCoefficientsFeatureExtractor):
        # super instances data
        super().setUp(dm_cls, extractor_cls)

    def tearDown(self):
        super().tearDown()

    def test_init(self):
        dm = self.dm
        extractor = self.extractor
        dm.load_all(cache=False)

        self.assertIsInstance(dm.X, np.ndarray)
        self.assertIsInstance(dm.Y, np.ndarray)

        self.assertEqual(1, dm.X.shape[1])  # channels is the second axis
        self.assertGreater(dm.X.shape[2], 1)  # w is the second axis
        self.assertGreater(dm.X.shape[3], 1)  # h is the second axis

        self.assertEqual(dm.sample_num, dm.X.shape[0])
        self.assertEqual(dm.sample_num, len(extractor.new_labels))
        self.assertEqual(dm.data_type, self.data_type)
        self.assertEqual(dm.feature_data_path, TEST_FEATURES_DATA_PATH / extractor.feature_name)

    def test_one_shot(self):
        dm = self.dm

        # load_all calls format_all who calls one_shot internally
        dm.load_all(cache=False)

        self.assertIsInstance(dm.Y, np.ndarray)
        self.assertEqual(len(dm.Y.shape), 1)
        self.assertEqual(dm.Y.shape[0], dm.sample_num)
        self.assertTrue((len(set(self.dm.Y)) > dm.Y).all())
        self.assertEqual(self.dm.Y.dtype, np.int64)

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

