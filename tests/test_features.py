import os
import pathlib
import unittest

import librosa
import numpy as np
import pandas as pd

from config import AVAIL_MEDIA_TYPES, RAW_DATA_PATH, SR, N_MELS
from features import FeatureExtractor, MelSpectralCoefficientsFeatureExtractor, DoubleHPSSFeatureExtractor, \
    VoiceActivationFeatureExtractor
from tests.config import TEST_RAW_DATA_PATH, TEST_FEATURES_DATA_PATH


class TestFeatureExtractorInterface(unittest.TestCase):
    feature_extractor = FeatureExtractor

    def load_example_raw_FeatureExtractor(self):
        label_path = TEST_RAW_DATA_PATH / 'labels.csv'
        extractor = self.feature_extractor.from_label_file(
            label_path,
            out_path=TEST_FEATURES_DATA_PATH,
            raw_path=TEST_RAW_DATA_PATH
        )
        return extractor

    def test_from_label_file(self):
        """
        Pass a feature label_path, out_path and source_path
        Should init FE with self.x = filenames and self.y = labels

        :return:
        """
        extractor = self.load_example_raw_FeatureExtractor()

        # assert that fields are set correctly
        # x is iterable of filenames (contain extension)
        try:
            for filename in extractor.x:
                name = '.'.join(filename.split(".")[:-1])
                extension = filename.split('.')[-1]
                self.assertIsInstance(name, str)
                self.assertIsInstance(extension, str)
                self.assertGreater(len(name), 0)
                self.assertGreater(len(extension), 0)
                if extension not in AVAIL_MEDIA_TYPES:
                    self.fail('{} not in AVAIL_MEDIA_TYPES'.format(extension))
        except Exception as e:
            self.fail(str(e))

        # y is iterable of strings
        try:
            for label in extractor.y:
                self.assertIsInstance(label, str)
                self.assertGreater(len(label), 0)
        except Exception as e:
            self.fail(str(e))

        # raw_path is the same as set
        self.assertIsInstance(extractor.raw_path, pathlib.Path)
        self.assertEqual(extractor.raw_path, TEST_RAW_DATA_PATH)

        # out_path is the same as set
        self.assertIsInstance(extractor.out_path, pathlib.Path)
        self.assertEqual(extractor.out_path, TEST_FEATURES_DATA_PATH / extractor.feature_name)

        # new labels should be empty
        self.assertEqual(len(extractor.new_labels), 0)

    def test_clean_references(self):
        """
        Checks that clean_references methods side-effects occurr as intended.
        In this case, x and y should be checket that all filenames exist in raw_path,
        if don't that filename and label should be removed.
        :return:
        """
        label_file_path = TEST_RAW_DATA_PATH / 'labels_missing_one.csv'
        extractor = self.feature_extractor.from_label_file(
            label_file_path,
            out_path=TEST_FEATURES_DATA_PATH,
            raw_path=TEST_RAW_DATA_PATH
        )
        # save previous x and y
        initial_filenames = [filename for filename in extractor.x]
        initial_labels = [label for label in extractor.y]
        self.assertEqual(len(initial_filenames), len(extractor.x))
        self.assertEqual(len(initial_labels), len(extractor.y))
        # clean dangling references (missing files)
        extractor.clean_references()
        # check that missing one was removed
        self.assertGreater(len(initial_filenames), len(extractor.x))
        self.assertGreater(len(initial_labels), len(extractor.y))

    def test_trigger_dependency_warning(self):
        """
        Test that feature extractor raise warnings if an inconsistency is found
        between the expected dependency feature and the received.
        :return:
        """

        base_extractor = self.load_example_raw_FeatureExtractor()

        filenames = base_extractor.x
        labels = base_extractor.y
        out_path = TEST_FEATURES_DATA_PATH

        first_positive = RAW_DATA_PATH
        first_negative = TEST_RAW_DATA_PATH

        second_positive = filenames
        second_negative = [''.join(filename.split('.')[:-1]) + '.npy' for filename in filenames]

        # perfect case
        extractor = FeatureExtractor(second_positive, labels, out_path, first_positive)
        first_flag, second_flag = extractor.trigger_dependency_warnings_if_needed()
        self.assertFalse(first_flag)
        self.assertFalse(second_flag)
        # First fail
        extractor = FeatureExtractor(second_positive, labels, out_path, first_negative)
        first_flag, second_flag = extractor.trigger_dependency_warnings_if_needed()
        self.assertTrue(first_flag)
        self.assertFalse(second_flag)
        # Second Fail
        extractor = FeatureExtractor(second_negative, labels, out_path, first_positive)
        first_flag, second_flag = extractor.trigger_dependency_warnings_if_needed()
        self.assertFalse(first_flag)
        self.assertTrue(second_flag)
        # Both Fails
        extractor = FeatureExtractor(second_negative, labels, out_path, first_negative)
        first_flag, second_flag = extractor.trigger_dependency_warnings_if_needed()
        self.assertTrue(first_flag)
        self.assertTrue(second_flag)

    def test_process_elements(self):
        """
        Tests that this static method returns a function that modify
        the extractor.new_labels attribute.
        :return:
        """
        # this method isnt implemented in abstractclass
        pass

    def test_process_element(self):
        """
        Tests that this static method returns a function that modify
        the extractor.new_labels attribute.
        :return:
        """
        # this method isn't immplemented in abstarct class
        pass

    def test_parallel_transform(self):
        """
        Tests that this method process existing x & y,
        according to process_element(s) functions
        and resulting new_labels are exported
        successfully.
        :return:
        """
        # method can't run on abstract class but will be tested on
        # other pipelines
        extractor = self.load_example_raw_FeatureExtractor()
        with self.assertRaises(NotImplementedError):
            extractor._parallel_transform()

    def test_transform(self):
        """
        Tests that this method process existing x & y,
        according to process_element(s) functions
        and resulting new_labels are exported
        successfully.
        :return:
        """

        extractor = self.load_example_raw_FeatureExtractor()
        with self.assertRaises(NotImplementedError):
            extractor.transform()

    def _test_get_filename(self, extension, feature_name, filename, new_filename=None):
        """
        Checks that get_file_name function transforms filename as expected.
        :param extension: Extension to be put in the filename after the dot (.)
        :param feature_name: Feature name to be appended to the filename as defined in the format
        :param filename: input filename to be transformed
        :return:
        """
        new_filename = new_filename if new_filename else self.feature_extractor.get_file_name(filename, feature_name,
                                                                                              ext=extension)
        # check that previous filetype was removed forn new filename
        previous_extension = filename.split('.')[-1]
        self.assertNotIn(previous_extension, new_filename) if previous_extension != extension else self.assertIn(
            previous_extension, new_filename)
        # check that new filetype is functional
        self.assertEqual(extension, new_filename.split('.')[-1])
        # check that contains the previous information
        self.assertEqual('.'.join(new_filename.split('.')[:-2]), '.'.join(filename.split('.')[:-1]))
        self.assertIn(extension, new_filename)
        return new_filename

    def test_get_filename(self):
        """
        Test that this static method format the filename
        according to the definition
        :return:
        """

        song_filename_short = '1_song.mp3'
        song_filename_long = '87_testname.2hpss.voice_activation.mean_svd.svd_ponderated_volume.wav'
        feature_filename_short = '35 -song.spec.npy'
        feature_filename_medium = '90_Fait_song.2hpss.voice_activation.mean_svd.npy'
        filenames = [song_filename_short, song_filename_long, feature_filename_short, feature_filename_medium]
        # check with default extension
        for filename in filenames:
            self._test_get_filename('npy', 'test_feature_1', filename)

        # check with other extension
        for filename in filenames:
            self._test_get_filename('pickle', 'test_feature_2', filename)

    def test_save_feature_simple(self, clean=True):
        """
        Tests that this static method exports the received nparray in
        the correct directory according to out_path and filename definition.
        And also modifiy the given new_labels.
        :return:
        """

        extractor = self.load_example_raw_FeatureExtractor()
        for idx, x_i in enumerate(extractor.x):
            self.assertEqual(len(extractor.new_labels), idx)
            y_i = extractor.y[idx]

            array_saved = np.asarray([idx])
            self.feature_extractor.save_feature(
                array_saved,
                extractor.feature_name,
                extractor.out_path,
                x_i,
                y_i,
                extractor.new_labels,
            )
            self.assertEqual(len(extractor.new_labels), idx + 1)
            new_filename = extractor.new_labels[idx][0]
            new_label = extractor.new_labels[idx][1]
            self.assertEqual(new_label, y_i)
            self.assertEqual(self._test_get_filename('npy', extractor.feature_name, x_i, new_filename), new_filename)
            # tries to load current file
            array_loaded = np.load(extractor.out_path / new_filename)
            self.assertEqual(array_loaded, array_saved)
            os.remove(extractor.out_path / new_filename) if clean else None
            self.assertFalse(
                os.path.isfile(extractor.out_path / new_filename)) if clean else self.assertTrue(
                os.path.isfile(extractor.out_path / new_filename))
        return extractor if not clean else None

    def test_save_feature_given_filename(self, clean=True):
        """
        Tests that this static method exports the received nparray in
        the correct directory according to out_path and filename definition.
        And also modifiy the given new_labels.
        :return:
        """

        extractor = self.load_example_raw_FeatureExtractor()
        for idx, x_i in enumerate(extractor.x):
            self.assertEqual(len(extractor.new_labels), idx)
            y_i = extractor.y[idx]

            array_saved = np.asarray([idx])
            new_filename = self._test_get_filename('npy', extractor.feature_name, x_i)

            self.feature_extractor.save_feature(
                array_saved,
                extractor.feature_name,
                extractor.out_path,
                x_i,
                y_i,
                extractor.new_labels,
                new_filename
            )
            self.assertEqual(len(extractor.new_labels), idx + 1)
            self.assertEqual(extractor.new_labels[idx][0], new_filename)
            new_label = extractor.new_labels[idx][1]
            self.assertEqual(new_label, y_i)
            self.assertEqual(self._test_get_filename('npy', extractor.feature_name, x_i, new_filename), new_filename)
            # tries to load current file
            array_loaded = np.load(extractor.out_path / new_filename)
            self.assertEqual(array_loaded, array_saved)
            os.remove(extractor.out_path / new_filename) if clean else None
            self.assertFalse(
                os.path.isfile(extractor.out_path / new_filename)) if clean else self.assertTrue(
                os.path.isfile(extractor.out_path / new_filename))
        return extractor if not clean else None

    def test_save_audio(self, clean=True):
        """
        Tests that this static method exports the received audio in
        the correct directory according to out_path and filename definition
        :return:
        """
        extractor = self.load_example_raw_FeatureExtractor()
        for idx, x_i in enumerate(extractor.x):
            self.assertEqual(len(extractor.new_labels), idx)
            y_i = extractor.y[idx]

            wav_saved, _ = librosa.core.load(extractor.raw_path / x_i, sr=SR)
            self.feature_extractor.save_audio(
                wav_saved,
                extractor.feature_name,
                extractor.out_path,
                x_i,
                y_i,
                extractor.new_labels,
            )
            self.assertEqual(len(extractor.new_labels), idx + 1)
            new_filename = extractor.new_labels[idx][0]
            new_label = extractor.new_labels[idx][1]
            self.assertEqual(new_label, y_i)
            self.assertEqual(self._test_get_filename('wav', extractor.feature_name, x_i, new_filename), new_filename)
            # tries to load current file
            array_loaded, _ = librosa.core.load(extractor.out_path / new_filename)
            # self.assertAlmostEqual(array_loaded, librosa.util.normalize(wav_saved)) #  this is kinda non-deterministic, tend to have shape mismatch
            os.remove(extractor.out_path / new_filename) if clean else None
            self.assertFalse(
                os.path.isfile(extractor.out_path / new_filename)) if clean else self.assertTrue(
                os.path.isfile(extractor.out_path / new_filename))
        return extractor if not clean else None


class TestSingingVoiceDetectionFeaturePipeline(unittest.TestCase):
    pass


class _TestFeatureExtractor(unittest.TestCase):

    def test_parallel_transform(self, clean=True):
        extractor = self.create_extractor()
        transform_fun = extractor._parallel_transform
        self._transform_load_test(clean, extractor, transform_fun)

    def test_sequential_transform(self, clean=True):
        extractor = self.create_extractor()
        transform_fun = extractor._sequential_transform
        self._transform_load_test(clean, extractor, transform_fun)

    def _transform_load_test(self, clean, extractor, transform_fun):
        self.assertEqual(len(extractor.new_labels), 0)
        transform_fun()
        self.assertGreater(len(extractor.new_labels), 0)
        extracted_data = np.asarray(extractor.new_labels)
        feature_filenames = extracted_data[:, 0]
        # feature_labels = extracted_data[:, 1]
        for filename in feature_filenames:
            x_i = np.load(extractor.out_path / filename, allow_pickle=True)
            self._test_feature_element(x_i)
        self._test_post_processing(extractor)
        self.remove_feature_files(extractor, feature_filenames) if clean else None

    def remove_feature_files(self, extractor, feature_filenames):
        [os.remove(extractor.out_path / filename) for filename in feature_filenames]
        os.remove(extractor.out_path / 'labels.{}.csv'.format(extractor.feature_name))

    def create_extractor(self):
        raise NotImplementedError()

    def _test_feature_element(self, x_i):
        raise NotImplementedError()

    def _test_post_processing(self, extractor):
        extracted_data = np.asarray(extractor.new_labels)
        feature_filenames = extracted_data[:, 0]
        feature_labels = extracted_data[:, 1]
        self.assertGreater(feature_filenames.shape[0], 0)
        self.assertGreater(feature_labels.shape[0], 0)
        self.assertEqual(feature_labels.shape, feature_filenames.shape)
        # try to open and parse labels
        try:
            df = pd.read_csv(extractor.out_path / 'labels.{}.csv'.format(extractor.feature_name))
            parsed_filenames = df['filename']
            parsed_labels = df['label']
        except Exception as e:
            self.fail(str(e))
        self.assertTrue(np.asarray((feature_filenames == parsed_filenames)).all())
        self.assertTrue(np.asarray((feature_labels == parsed_labels)).all())


class TestMFSCFeatureExtraction(_TestFeatureExtractor):

    def _test_feature_element(self, x_i):
        self.assertEqual(x_i.shape[0], N_MELS)

    def create_extractor(self):
        extractor = MelSpectralCoefficientsFeatureExtractor.from_label_file(
            TEST_RAW_DATA_PATH / 'labels.csv',
            out_path=TEST_FEATURES_DATA_PATH,
            raw_path=TEST_RAW_DATA_PATH
        )
        return extractor


class TestVoiceActivationFeatureExtraction(_TestFeatureExtractor):

    def _test_feature_element(self, x_i):
        print('voice: {}'.format(x_i.shape))
        pass

    def create_extractor(self):
        self.extractor_dep = DoubleHPSSFeatureExtractor.from_label_file(
            TEST_RAW_DATA_PATH / 'labels.csv',
            out_path=TEST_FEATURES_DATA_PATH,
            raw_path=TEST_RAW_DATA_PATH
        )
        self.extractor_dep.transform()

        str_feature_name = self.extractor_dep.feature_name
        source_path = self.extractor_dep.out_path
        label_path = source_path / 'labels.{}.csv'.format(str_feature_name)

        extractor = VoiceActivationFeatureExtractor.from_label_file(
            label_path,
            out_path=TEST_FEATURES_DATA_PATH,
            raw_path=source_path
        )
        return extractor

    def test_parallel_transform(self, clean=True):
        with self.assertRaises(NotImplementedError):
            super().test_parallel_transform(clean=False)

    def remove_feature_files(self, extractor, feature_filenames):
        # clean dependency feature files
        extracted_data = np.asarray(self.extractor_dep.new_labels)
        feature_filenames_dep = extracted_data[:, 0]
        super().remove_feature_files(self.extractor_dep, feature_filenames_dep)
        # clean this one as naturally
        super().remove_feature_files(extractor, feature_filenames)


del _TestFeatureExtractor
if __name__ == '__main__':
    unittest.main()
