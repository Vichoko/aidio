class CNNTrainer(AbstractTrainer):
    """
    Trainer for a simple class classification CNN.
    """

    def __init__(self, dataset_name, train_validation_split=.8, resume_checkpoint=None, batch_size=16, workers=4,
                 n_gpu=0, epochs=2, learning_rate=.01, momentum=.8):
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(dataset_name, train_validation_split, resume_checkpoint, batch_size, workers, n_gpu, epochs)

    @property
    def initial_model(self):
        return ClassificationConvolutionalNetwork()

    @property
    def loss(self):
        return CrossEntropyLoss()

    @property
    def optimizer(self):
        return SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'momentum': self.momentum}

    @property
    def trainer_id(self):
        return 'cnn_sk'

    def _create_evaluator_engine(self):
        return create_supervised_evaluator(
            self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)}, device=self.device)

    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)


if __name__ == '__main__':
    trainer = CNNTrainer(dataset_name, train_validation_split, resume_checkpoint, batch_size, workers, n_gpu, epochs,
                         learning_rate, momentum)
    trainer.run()
