class ClassificationModel:
    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        return

    def data_loader(self, audio_data, label_data=None):
        raise NotImplemented

    def train(self, audio_data, label_data, options):
        """
        Train model specified by options with given data.

        :param audio_data: iterable reference
        :param label_data: iterable reference
        :param options: dict-like; model dependant (cnn, aidsan, etc)
        :return:
        """
        self.x, self.y = self.data_loader(audio_data, label_data)
        return

    def predict(self, audio_data, options):
        """
        Predict with given data, and options.

        :param audio_data: reference
        :param options:
        :return:
        """
        self.x = self.data_loader(audio_data)
        return


class AudioProcessor:
    def data_loader(self, data):
        """
        load iterable of references to actual manegable objects
        :param data:
        :return:
        """
        raise NotImplemented

    def transform(self, data, options):
        """

        :param data: iterable of references
        :param options:
        :return:
        """
        return



