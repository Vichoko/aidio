import pathlib

from config import makedirs

TEST_STATIC_FILES_PATH = pathlib.Path('./tests/static/')
TEST_RAW_DATA_PATH = TEST_STATIC_FILES_PATH / 'data/raw/'
TEST_FEATURES_DATA_PATH = TEST_STATIC_FILES_PATH / 'data/features/'
TEST_DIGEST_DATA_PATH = TEST_STATIC_FILES_PATH / 'data/digest/'
TEST_MODELS_DATA_PATH = TEST_STATIC_FILES_PATH / 'data/models/'


makedirs(TEST_STATIC_FILES_PATH)
makedirs(TEST_RAW_DATA_PATH)
makedirs(TEST_FEATURES_DATA_PATH)
makedirs(TEST_DIGEST_DATA_PATH)
makedirs(TEST_MODELS_DATA_PATH)
