from interfaces import ClassificationModel


class ResNet18(ClassificationModel):
    model_name = 'resnet18'

    def __init__(self):
        super().__init__(self.model_name)

