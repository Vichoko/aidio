from argparse import ArgumentParser
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as ptl
import torch
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision.models import resnext50_32x4d

from config import WAVENET_BATCH_SIZE, NUM_WORKERS, RESNET_V2_BATCH_SIZE
from loaders import ClassSampler
from torch_models import WaveNetTransformerClassifier, GMMClassifier, WaveNetBiLSTMClassifier, WaveNetLSTMClassifier


class L_GMMClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_GMMClassifier, self).__init__()
        self.hparams = hparams
        # self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = GMMClassifier(num_classes)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)

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
        x, y = batch['x'], batch['y']

        for gmm_idx, gmm in enumerate(self.model.gmm_list):
            gmm.fit()

        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
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
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
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

    # def __dataloader(self, train):
    #     # init data generators
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.5,), (1.0,))])
    #     dataset = MNIST(root=self.hparams.data_root, train=train,
    #                     transform=transform, download=True)
    #
    #     # when using multi-node (ddp) we need to add the  datasampler
    #     train_sampler = None
    #     batch_size = self.hparams.batch_size
    #
    #     if self.use_ddp:
    #         train_sampler = DistributedSampler(dataset)
    #
    #     should_shuffle = train_sampler is None
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=0
    #     )
    #
    #     return loader

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(self.train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(self.eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(self.test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=WAVENET_BATCH_SIZE, type=int)
        return parser


class L_WavenetTransformerClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_WavenetTransformerClassifier, self).__init__()
        self.hparams = hparams
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = WaveNetTransformerClassifier(num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)

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
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
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
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
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

    # def __dataloader(self, train):
    #     # init data generators
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.5,), (1.0,))])
    #     dataset = MNIST(root=self.hparams.data_root, train=train,
    #                     transform=transform, download=True)
    #
    #     # when using multi-node (ddp) we need to add the  datasampler
    #     train_sampler = None
    #     batch_size = self.hparams.batch_size
    #
    #     if self.use_ddp:
    #         train_sampler = DistributedSampler(dataset)
    #
    #     should_shuffle = train_sampler is None
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=0
    #     )
    #
    #     return loader

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(self.train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(self.eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(self.test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=WAVENET_BATCH_SIZE, type=int)
        parser.add_argument('--gpus', default=0, type=int)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_WavenetLSTMClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_WavenetLSTMClassifier, self).__init__()
        self.hparams = hparams
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = WaveNetLSTMClassifier(num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)

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
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
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
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
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

    # def __dataloader(self, train):
    #     # init data generators
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.5,), (1.0,))])
    #     dataset = MNIST(root=self.hparams.data_root, train=train,
    #                     transform=transform, download=True)
    #
    #     # when using multi-node (ddp) we need to add the  datasampler
    #     train_sampler = None
    #     batch_size = self.hparams.batch_size
    #
    #     if self.use_ddp:
    #         train_sampler = DistributedSampler(dataset)
    #
    #     should_shuffle = train_sampler is None
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=0
    #     )
    #
    #     return loader

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(self.train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(self.eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(self.test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=WAVENET_BATCH_SIZE, type=int)
        parser.add_argument('--gpus', default=0, type=int)
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
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = resnext50_32x4d(num_classes=num_classes)
        input_channels = 1
        initial_inplanes = 64
        self.model.conv1 = Conv2d(input_channels, initial_inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)

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
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_pred, y)

        # acc
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
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
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
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

    # def __dataloader(self, train):
    #     # init data generators
    #     transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize((0.5,), (1.0,))])
    #     dataset = MNIST(root=self.hparams.data_root, train=train,
    #                     transform=transform, download=True)
    #
    #     # when using multi-node (ddp) we need to add the  datasampler
    #     train_sampler = None
    #     batch_size = self.hparams.batch_size
    #
    #     if self.use_ddp:
    #         train_sampler = DistributedSampler(dataset)
    #
    #     should_shuffle = train_sampler is None
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=0
    #     )
    #
    #     return loader

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(self.train_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(self.eval_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(self.test_dataset, batch_size=WAVENET_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=RESNET_V2_BATCH_SIZE, type=int)
        parser.add_argument('--gpus', default=0, type=int)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser