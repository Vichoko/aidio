"""
End to end proccesses defined by a sequence of procedures of the same kind
"""
from pathlib import Path

from config import FEATURES_DATA_PATH, RAW_DATA_PATH
from features import DoubleHPSSFeatureExtractor, VoiceActivationFeatureExtractor, MeanSVDFeatureExtractor, \
    SVDPonderatedVolumeFeatureExtractor, IntensitySplitterFeatureExtractor


class FeatureExtractionPipeline:
    def __init__(self, feature_path=FEATURES_DATA_PATH, raw_path=RAW_DATA_PATH):
        self.feature_path = feature_path
        self.raw_path = raw_path
        self.pipeline = []
        self.instanced_extractors = []

    def execute(self):
        if not self.pipeline:
            raise NotImplementedError('self.pipeline is undefined: {}'.format(self.pipeline))

        for step_proccess in self.pipeline:
            print('info: initializing transform: {}'.format(step_proccess))
            extractor = step_proccess.magic_init(self.feature_path, self.raw_path)
            extractor.transform()
            self.instanced_extractors.append(extractor)
            print('info: transformed!')


class SVDPipeline(FeatureExtractionPipeline):

    def __init__(self, feature_path, raw_path):
        super().__init__(feature_path, raw_path)
        self.pipeline = [
            DoubleHPSSFeatureExtractor,
            VoiceActivationFeatureExtractor,
            MeanSVDFeatureExtractor,
            SVDPonderatedVolumeFeatureExtractor,
            IntensitySplitterFeatureExtractor
        ]


if __name__ == '__main__':
    p = SVDPipeline(Path('E:/aidio_data/features'), Path('E:/parsed_singers.v2'))
    p.execute()
