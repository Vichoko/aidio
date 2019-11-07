__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Abstract class with the boilerplate code needed to define and run an Ignite engine. """


import os
import torch

from datetime import datetime
from ignite.engine import Events
from ignite.handlers import Timer, ModelCheckpoint, TerminateOnNan
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import get_device, get_checkpoint_directory, get_log_directory
from src.utils.data import dataset_split_successive


class AbstractTrainer:
    """
    Abstract class with the boilerplate code needed to define and run an Ignite trainer Engine.
    """
    def __init__(self, dataset_name, train_validation_split=.8, resume_date=None, batch_size=16, workers=4,
                 n_gpu=0, epochs=2):
        date = resume_date if resume_date else datetime.now()
        self.device = get_device(n_gpu)
        self.model = self.initial_model.to(self.device)
        self.dataset = get_dataset(dataset_name)
        self.dataset_name = dataset_name
        self.train_loader, self.val_loader = self._create_data_loaders(train_validation_split, batch_size, workers)
        self.log_directory = get_log_directory(self.trainer_id, date=date)
        self.checkpoint_directory = get_checkpoint_directory(self.trainer_id, date=date)
        self.evaluator = self._create_evaluator_engine()
        self.trainer_engine = self._create_trainer_engine()
        self.batch_size = batch_size
        self.start_epoch = 0
        self.epochs = epochs
        self.timer = self._create_timer()
        self.event_handlers = []
        self._add_event_handlers()

    @property
    def initial_model(self):
        """
        Getter for an untrained nn.module for the model the Trainer is designed for. Used to initialize `self.model`.
        :return: a model object
        :type: torch.nn.module
        """
        raise NotImplementedError

    @property
    def loss(self):
        """
        Getter for the loss to be used during training.
        :return: a loss object
        :type: torch.nn._Loss
        """
        raise NotImplementedError

    @property
    def optimizer(self):
        """
        Getter for the optimizer to be used during training.
        :return: an optimizer object
        :type: torch.optim.Optimizer
        """
        raise NotImplementedError

    @property
    def serialized_checkpoint(self):
        """
        Getter for the serialized checkpoint dictionary, which contains the values of the trainer's fields that should
        be saved in a trainer checkpoint.
        :return: a checkpoint dictionary
        :type: dict
        """
        return {
            'dataset_name': self.dataset_name,
            'total_epochs': self.start_epoch + self.epochs,
            'batch_size': self.batch_size,
            'model': self.initial_model,
            'optimizer': self.optimizer,
            'last_run': datetime.now(),
            'average_epoch_duration': self.timer.value()
        }

    @property
    def trainer_id(self):
        """
        Getter for the trainer id, a unique str to identify the trainer. The corresponding `data` directory sub-folders
        will get a name containing this id.
        :return: the trainer id
        :type: str
        """
        raise NotImplementedError

    def _create_evaluator_engine(self):
        """
        Creates an Ignite evaluator engine for the target model.
        :return: an evaluator engine for the target model
        :type: ignite.Engine
        """
        raise NotImplementedError

    def _create_trainer_engine(self):
        """
        Creates an Ignite training engine for the target model.
        :return: a trainer engine for the target model
        :type: ignite.Engine
        """
        raise NotImplementedError

    def _deserialize_checkpoint(self, checkpoint):
        """
        Load trainer fields from a serialized checkpoint dictionary.
        :param checkpoint: the checkpoint being loaded
        :type: dict
        """
        self.start_epoch = checkpoint['epochs']
        self.model = torch.load(os.path.join(self.checkpoint_directory, '_model_{}.pth'.format(self.start_epoch)))

    def _load_checkpoint(self, resume_date):
        """
        Load the trainer checkpoint dictionary for the given resume date and deserialize it (load the values in the
        checkpoint dictionary into the trainer's fields).
        :param resume_date: checkpoint folder name containing model and checkpoint .pth files containing the information
        needed for resuming training. Folder names correspond to dates with the following format: `%y-%m-%dT%H-%M`
        :type: str
        """
        try:
            self._deserialize_checkpoint(torch.load(os.path.join(self.checkpoint_directory, 'checkpoint.pth')))
            print('Successfully loaded the {} checkpoint.'.format(resume_date))
        except FileNotFoundError:
            raise FileNotFoundError('Checkpoint {} not found.'.format(resume_date))

    def _save_checkpoint(self):
        """
        Create the serialized checkpoint dictionary for the current trainer state, and save it.
        """
        torch.save(self.serialized_checkpoint, os.path.join(self.checkpoint_directory, 'checkpoint.pth'))

    def _add_event_handlers(self):
        """
        Adds a progressbar and a summary writer to output the current training status. Adds event handlers to output
        common messages and update the progressbar.
        """
        progressbar_description = 'TRAINING => loss: {:.6f}'
        progressbar = tqdm(initial=0, leave=False, total=len(self.train_loader), desc=progressbar_description.format(0))
        writer = SummaryWriter(self.log_directory)

        @self.trainer_engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            writer.add_scalar('loss', trainer.state.output)
            progressbar.desc = progressbar_description.format(trainer.state.output)
            progressbar.update(1)

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            progressbar.n = progressbar.last_print_n = 0
            self.evaluator.run(self.train_loader)
            metrics = self.evaluator.state.metrics
            for key, value in metrics.items():
                writer.add_scalar(key, value)
            tqdm.write('\nTraining Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}\n'
                       .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            progressbar.n = progressbar.last_print_n = 0
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            for key, value in metrics.items():
                writer.add_scalar(key, value)
            tqdm.write('\nValidation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}\n'
                       .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))

        checkpoint_saver = ModelCheckpoint( # create a Checkpoint handler that can be used to periodically
            self.checkpoint_directory, filename_prefix='net', # save model objects to disc.
            save_interval=1, n_saved=5, atomic=True, create_dir=True, save_as_state_dict=False, require_empty=False
        )
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, {'train': self.model})
        self.trainer_engine.add_event_handler(Events.COMPLETED, checkpoint_saver, {'complete': self.model})
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def _create_data_loaders(self, train_validation_split, batch_size, workers):
        """
        Create training and validation data loaders, placing a total of `len(self.dataset) * train_validation_split`
        elements in the training subset.
        :param train_validation_split: the proportion of dataset elements to be placed in the training subset.
        :type: float $\in [0, 1]$
        :param batch_size: batch size for both data loaders.
        :type: int
        :param workers: number of workers for both data loaders.
        :type: int
        :return: two DataLoaders, the first for the training data and the second for the validation data.
        :type: torch.utils.data.DataLoader
        """
        return [DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=workers)
                for subset in dataset_split_successive(self.dataset, train_validation_split)]

    def _create_timer(self):
        """
        Create and attach a new timer to the trainer, registering callbacks.
        :return: the newly created timer
        :type: ignite.handlers.Timer
        """
        timer = Timer(average=True)
        timer.attach(self.trainer_engine, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        return timer

    def run(self):
        """
        Run the trainer.
        """
        self.trainer_engine.run(self.train_loader, max_epochs=self.epochs)
