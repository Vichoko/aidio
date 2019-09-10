import unittest

from models import ResNetV2


class TestModel(unittest.TestCase):
    def test_init(self):
        pass
        # model = ResNetV2()
        # for x_train, y_train, x_test, y_test in model.get_shuffle_split():
        #     model.train(x_train, y_train, x_test, y_test)
        #     model.evaluate(x_test, y_test)

