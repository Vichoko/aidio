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

from config import WAVENET_BATCH_SIZE, NUM_WORKERS, RESNET_V2_BATCH_SIZE, SR, WAVENET_LEARNING_RATE, \
    WAVENET_WEIGHT_DECAY, WAVENET_USE_AMSGRAD
from loaders import ClassSampler
from torch_models import WaveNetTransformerClassifier, GMMClassifier, WaveNetLSTMClassifier, WaveNetClassifier


class DummyOptimizer(torch.optim.Optimizer):
    pass


class L_AbstractClassifier(ptl.LightningModule):
    """
        Sample model to show how to define a template
        """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, torch_module):
        super(L_AbstractClassifier, self).__init__()
        self.hparams = hparams
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = torch_module
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
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_GMMClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_GMMClassifier, self).__init__()
        self.model_path = None
        self.num_classes = num_classes
        self.hparams = hparams
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.optimizer = torch.optim.Adam([torch.Tensor()], lr=hparams.learning_rate)
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
        val_dataloader = self.val_dataloader()[0]
        test_dataloader = self.test_dataloader()[0]
        val_out = []
        for batch_idx, batch in tqdm.tqdm(enumerate(val_dataloader)):
            val_out.append(self.validation_step(batch, batch_idx))
        res = self.validation_end(val_out)
        print(res['log'])
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
        print('info: Getting train data metrics...')
        # y_pred = self.forward(x)
        # # as torch methods expect first dim to be N, add first dimension to 1
        # # calculate loss
        # loss_val = self.loss(y_pred, y)
        # tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            # 'loss': loss_val,
            # 'progress_bar': tqdm_dict,
            # 'log': tqdm_dict
        })
        print('info: Done!')
        # can also return just a scalar instead of a dict (return loss_val)
        return output

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
            val_acc = output['val_acc']
            val_loss_mean += val_loss
            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {
            'val_loss': val_loss_mean,
            'val_acc': val_acc_mean
        }
        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            # 'val_loss': val_loss_mean
        }
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

    @ptl.data_loader
    def train_dataloader(self):
        # logging.info('training data loader called')
        return DataLoader(
            self.train_dataset,
            num_workers=NUM_WORKERS,
            batch_sampler=ClassSampler(self.num_classes, self.train_dataset.labels),
        )

    @ptl.data_loader
    def val_dataloader(self):
        # logging.info('val data loader called')
        return DataLoader(
            self.eval_dataset,
            num_workers=NUM_WORKERS,
            batch_sampler=ClassSampler(self.num_classes, self.eval_dataset.labels),
        )

    @ptl.data_loader
    def test_dataloader(self):
        # logging.info('test data loader called')
        return DataLoader(
            self.test_dataset,
            num_workers=NUM_WORKERS,
            batch_sampler=ClassSampler(self.num_classes, self.test_dataset.labels),
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
        parser.add_argument('--learning_rate', default=0.001, type=float)
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
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate,
                                          weight_decay=hparams.weight_decay, amsgrad=WAVENET_USE_AMSGRAD)

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
        # summary(self.model, input_size=(1, SR * 10), device="cpu")
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
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser


class L_WavenetClassifier(ptl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super(L_WavenetClassifier, self).__init__()
        self.hparams = hparams
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        # build model
        self.model = WaveNetClassifier(num_classes)
        summary(self.model, input_size=(1, SR * 10), device="cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate,
                                          weight_decay=hparams.weight_decay)

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
