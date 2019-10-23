import pathlib
import unittest

import librosa
import numpy as np
import pandas as pd
from builtins import NotImplementedError

from config import SR
from features import IntensitySplitterFeatureExtractor
from pipelines import FeatureExtractionPipeline, SVDPipeline
from tests.config import TEST_RAW_DATA_PATH, TEST_FEATURES_DATA_PATH


class _TestPipelineFeatureExtraction(unittest.TestCase):
    pipeline = FeatureExtractionPipeline

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.pipeline = cls.pipeline(feature_path=TEST_FEATURES_DATA_PATH, raw_path=TEST_RAW_DATA_PATH)

    def test_init(self):
        """
        Pass a feature label_path, out_path and source_path
        Should init FE with self.x = filenames and self.y = labels

        :return:
        """
        pipeline = self.pipeline

        # assert that fields are set correctly
        # raw_path is the same as set
        self.assertIsInstance(pipeline.raw_path, pathlib.Path)
        self.assertEqual(pipeline.raw_path, TEST_RAW_DATA_PATH)

        # out_path is the same as set
        self.assertIsInstance(pipeline.feature_path, pathlib.Path)
        self.assertEqual(pipeline.feature_path, TEST_FEATURES_DATA_PATH)


class TestPipelineFeatureExtractorInterface(_TestPipelineFeatureExtraction):
    def test_init(self):
        super().test_init()
        # new labels should be empty
        self.assertEqual(len(self.pipeline.pipeline), 0)

    def test_execute(self):
        with self.assertRaises(NotImplementedError):
            self.pipeline.execute()


class TestSingingVoiceDetectionFeaturePipeline(_TestPipelineFeatureExtraction):
    pipeline = SVDPipeline

    def test_init(self):
        super().test_init()
        # new labels should be empty
        self.assertEqual(len(self.pipeline.pipeline), 5)

    def test_execute(self):
        clean = True
        self.assertEqual(len(self.pipeline.instanced_extractors), 0)
        self.pipeline.execute()
        self.assertGreater(len(self.pipeline.instanced_extractors), 0)
        self.assertEqual(len(self.pipeline.instanced_extractors), len(self.pipeline.pipeline))

        for extractor in self.pipeline.instanced_extractors:
            if isinstance(extractor, IntensitySplitterFeatureExtractor):
                continue
            self.assertGreater(len(extractor.new_labels), 0)
            extracted_data = np.asarray(extractor.new_labels)
            feature_filenames = extracted_data[:, 0]
            # feature_labels = extracted_data[:, 1]
            for filename in feature_filenames:
                if 'npy' in filename:
                    x_i = np.load(extractor.out_path / filename, allow_pickle=True)
                else:
                    x_i = librosa.load(extractor.out_path / filename, sr=SR)

                # self._test_feature_element(x_i)
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
            extractor.remove_feature_files(feature_filenames) if clean else None


del _TestPipelineFeatureExtraction
if __name__ == '__main__':
    unittest.main()
