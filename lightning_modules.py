import pickle
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as ptl
import torch
import tqdm
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.models import resnext50_32x4d

from config import WAVENET_BATCH_SIZE, DATA_LOADER_NUM_WORKERS, RESNET_V2_BATCH_SIZE, WAVENET_LEARNING_RATE, \
    WAVENET_WEIGHT_DECAY, WNTF_BATCH_SIZE, WNLSTM_BATCH_SIZE, WAVEFORM_MAX_SEQUENCE_LENGTH, GMM_PREDICT_BATCH_SIZE, \
    GMM_TRAIN_BATCH_SIZE, CONV1D_LEARNING_RATE, CONV1D_WEIGHT_DECAY, CONV1D_BATCH_SIZE, WNTF_LEARNING_RATE, \
    WNTF_WEIGHT_DECAY, WNLSTM_LEARNING_RATE, WNLSTM_WEIGHT_DECAY, RNN1D_BATCH_SIZE, RNN1D_LEARNING_RATE, \
    RNN1D_WEIGHT_DECAY, RESNET_V2_LR
from loaders import ClassSampler
from torch_models import WaveNetTransformerClassifier, GMMClassifier, WaveNetClassifier, \
    Conv1DClassifier, RNNClassifier, WaveNetLSTMClassifier


class DummyOptimizer(torch.optim.Optimizer):
    pass


class L_GMMClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_GMMClassifier, self).__init__()
        self.model_path = None
        self.num_classes = num_classes
        self.hparams = hparams
        self.predict_batch_size = hparams.predict_batch_size
        self.train_batch_size = hparams.train_batch_size
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.optimizer = DummyOptimizer([torch.Tensor()], {})
        self.trained = False
        self.model = None
        # self.model = self.load_model(self.num_classes)

    def train_now(self):
        if self.trained:
            print('warning: Trying to train an alredy fitted GMM loaded from folder: {}. Skipping train...'.format(
                self.model_path))
            return -1

        print('info: starting training')
        train_dataloader = self.train_dataloader()
        for batch_idx, batch in tqdm.tqdm(enumerate(train_dataloader)):
            self.training_step(batch, batch_idx)
        self.trained = True
        print('info: ending training')
        return 0

    def eval_now(self):
        print('info: starting evaluation')
        val_dataloader = self.val_dataloader()
        test_dataloader = self.test_dataloader()
        val_out = []
        for batch_idx, batch in tqdm.tqdm(enumerate(val_dataloader)):
            val_out.append(self.validation_step(batch, batch_idx))
        res = self.validation_end(val_out)
        print(res)
        print('info: ending evaluation')
        return 0

    def save_model(self, model_path):
        """
        Save the model state with some metrics.

        :return:
        """
        filename = 'model.pickle'
        pickle.dump(self.model, open(model_path / filename, 'wb'))
        print('info: gmm model saved')
        return

    def load_model(self, model_path):
        filename = 'model.pickle'
        self.model_path = model_path
        try:
            model = pickle.load(open(model_path / filename, 'rb'))
            print('info: gmm loaded from file')
            self.trained = True
        except IOError:
            model = GMMClassifier(self.num_classes)
            print('info: previuous gmm not found.')
            self.trained = False
        self.model = model
        return model

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        debug = True
        x, y = batch['x'], batch['y']
        print('debug: batch is {}, labels are {}'.format(batch_idx, y)) if debug else None
        self.model.fit(x, y)
        result = ptl.TrainResult()
        return result

    def backward(self, use_amp, loss, optimizer):
        return

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # as torch methods expect first dim to be N, add first dimension to 1
        # calculate loss
        loss_val = self.loss(y_pred, y)
        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        # if self.on_gpu:
        #     val_acc = val_acc.cuda(loss_val.device.index)
        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
            'meta_data': {
                'data_count': len(x),
                'data_shape': x.shape
            }
        })
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        data_count = 0

        for output in outputs:
            val_loss = output['val_loss']
            val_acc = output['val_acc']
            val_loss_mean += val_loss
            val_acc_mean += val_acc
            data_count += output['meta_data']['data_count']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {
            'val_loss': val_loss_mean,
            'val_acc': val_acc_mean
        }
        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'hiddens': {
                'total_data_count': data_count,
                'last_point_metadata': outputs[-1]['meta_data']
            }

        }
        return result

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop
        :param batch:
        :return:
        """
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        """
        Called at the end of test to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        print('info: Testing complete.')
        result = self.validation_end(outputs)
        print('info: {}'.format(result['log']))
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [self.optimizer]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=DATA_LOADER_NUM_WORKERS,
            batch_sampler=ClassSampler(self.num_classes, self.train_dataset.labels, self.train_batch_size),
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.predict_batch_size,
            num_workers=DATA_LOADER_NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.predict_batch_size,
            num_workers=DATA_LOADER_NUM_WORKERS
        )

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--predict_batch_size', default=GMM_PREDICT_BATCH_SIZE, type=float)
        parser.add_argument('--train_batch_size', default=GMM_TRAIN_BATCH_SIZE, type=float)
        return parser


class L_WavenetAbstractClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.wd = hparams.weight_decay
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate loss
        loss = self.loss(y_pred, y)
        result = ptl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate loss
        loss = self.loss(y_pred, y)
        # calculate accurracy
        labels_hat = torch.argmax(y_pred, dim=1)
        accuracy = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        accuracy = torch.tensor(accuracy)
        if self.on_gpu:
            accuracy = accuracy.cuda(loss.device.index)
        # Checkpoint model based on validation loss
        result = ptl.EvalResult(early_stop_on=None, checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', accuracy, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate loss
        loss = self.loss(y_pred, y)
        # calculate accurracy
        labels_hat = torch.argmax(y_pred, dim=1)
        accuracy = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        accuracy = torch.tensor(accuracy)
        if self.on_gpu:
            accuracy = accuracy.cuda(loss.device.index)
        # Checkpoint model based on validation loss
        result = ptl.EvalResult()
        result.log('test_loss', loss, prog_bar=True)
        result.log('test_acc', accuracy, prog_bar=True)
        return result

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [self.optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=DATA_LOADER_NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=DATA_LOADER_NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=DATA_LOADER_NUM_WORKERS)


class L_WavenetClassifier(L_WavenetAbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
        # build model
        self.model = WaveNetClassifier(num_classes)
        summary(self.model, input_size=(1, WAVEFORM_MAX_SEQUENCE_LENGTH), device="cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.wd)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=WAVENET_LEARNING_RATE, type=float)
        parser.add_argument('--weight_decay', default=WAVENET_WEIGHT_DECAY, type=float)
        parser.add_argument('--batch_size', default=WAVENET_BATCH_SIZE, type=int)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_Conv1DClassifier(L_WavenetAbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
        # build model
        self.model = Conv1DClassifier(num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.wd)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=CONV1D_LEARNING_RATE, type=float)
        parser.add_argument('--weight_decay', default=CONV1D_WEIGHT_DECAY, type=float)
        parser.add_argument('--batch_size', default=CONV1D_BATCH_SIZE, type=int)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_WavenetTransformerClassifier(L_WavenetAbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset)
        # build model
        self.model = WaveNetTransformerClassifier(num_classes)
        summary(self.model, input_size=(1, WAVEFORM_MAX_SEQUENCE_LENGTH), device="cpu")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            # amsgrad=WAVENET_USE_AMSGRAD
        )

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=WNTF_LEARNING_RATE, type=float)
        parser.add_argument('--weight_decay', default=WNTF_WEIGHT_DECAY, type=float)
        parser.add_argument('--batch_size', default=WNTF_BATCH_SIZE, type=int)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_WavenetLSTMClassifier(L_WavenetAbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
        # build model
        self.model = WaveNetLSTMClassifier(num_classes)
        # summary(self.model, input_size=(WNLSTM_BATCH_SIZE, 1, WAVEFORM_MAX_SEQUENCE_LENGTH), device="cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=WNLSTM_LEARNING_RATE, type=float)
        parser.add_argument('--batch_size', default=WNLSTM_BATCH_SIZE, type=int)
        parser.add_argument('--weight_decay', default=WNLSTM_WEIGHT_DECAY, type=float)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_RNNClassifier(L_WavenetAbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
        # build model
        self.model = RNNClassifier(num_classes)
        # summary(self.model, input_size=(RNN1D_BATCH_SIZE, 1, WAVEFORM_MAX_SEQUENCE_LENGTH), device="cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=RNN1D_LEARNING_RATE, type=float)
        parser.add_argument('--batch_size', default=RNN1D_BATCH_SIZE, type=int)
        parser.add_argument('--weight_decay', default=RNN1D_WEIGHT_DECAY, type=float)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_ResNext50(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_ResNext50, self).__init__()
        self.hparams = hparams
        self.lr = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = resnext50_32x4d(num_classes=num_classes)
        input_channels = 1
        self.model.conv1 = Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = x.unsqueeze(1).float()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate loss
        loss = self.loss(y_pred, y)
        result = ptl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate loss
        loss = self.loss(y_pred, y)
        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        accuracy = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        accuracy = torch.tensor(accuracy)
        if self.on_gpu:
            accuracy = accuracy.cuda(loss.device.index)
        result = ptl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', accuracy, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate loss
        loss = self.loss(y_pred, y)
        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        accuracy = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        accuracy = torch.tensor(accuracy)
        if self.on_gpu:
            accuracy = accuracy.cuda(loss.device.index)
        result = ptl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, prog_bar=True)
        result.log('test_acc', accuracy, prog_bar=True)
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [self.optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=DATA_LOADER_NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=DATA_LOADER_NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=DATA_LOADER_NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=RESNET_V2_LR, type=float)
        parser.add_argument('--batch_size', default=RESNET_V2_BATCH_SIZE, type=int)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser
