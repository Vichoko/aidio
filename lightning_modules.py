from argparse import ArgumentParser
from collections import OrderedDict
from os.path import isfile

import numpy as np
import pytorch_lightning as ptl
import torch
import tqdm
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.models import resnext50_32x4d

from config import WAVENET_BATCH_SIZE, DATA_LOADER_NUM_WORKERS, RESNET_V2_BATCH_SIZE, WAVENET_LEARNING_RATE, \
    WAVENET_WEIGHT_DECAY, WNTF_BATCH_SIZE, WNLSTM_BATCH_SIZE, WAVEFORM_RANDOM_CROP_SEQUENCE_LENGTH, \
    GMM_PREDICT_BATCH_SIZE, \
    GMM_TRAIN_BATCH_SIZE, CONV1D_LEARNING_RATE, CONV1D_WEIGHT_DECAY, CONV1D_BATCH_SIZE, WNTF_LEARNING_RATE, \
    WNTF_WEIGHT_DECAY, WNLSTM_LEARNING_RATE, WNLSTM_WEIGHT_DECAY, RNN1D_BATCH_SIZE, RNN1D_LEARNING_RATE, \
    RNN1D_WEIGHT_DECAY, RESNET_V2_LR, RESNET_V2_WEIGHT_DECAY
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
        self.model = GMMClassifier(self.num_classes)
        self.gmm_filename = 'gmm_{}_of_{}.pickle'
        self.trained_gmm_indices = None

    def start_training(self):
        if self.trained:
            print('warning: Trying to train an alredy fitted GMM loaded from folder: {}. Skipping train...'.format(
                self.model_path))
            return -1
        print('info: starting training')
        print('info: skipping load and train of already trained GMMs {} found. '.format(
            self.trained_gmm_indices)) if self.trained_gmm_indices else None

        batch_generator = self.train_dataloader()
        print('info: starting batch data loading...')
        for batch_idx, batch in tqdm.tqdm(enumerate(batch_generator()), desc='Batch', unit='#',
                                          total=self.num_classes):
            if batch_idx in self.trained_gmm_indices and batch is None:
                print('info: skipping batch {} already trained.')
                continue
            assert batch_idx not in self.trained_gmm_indices and batch is not None, 'error: inconsitency in batch skipping mechanism.'
            print('info: batch {} loaded!'.format(batch_idx))
            self.training_step(batch, batch_idx)
            self.model.save_gmm(
                batch_idx,
                self.model_path / self.gmm_filename.format(batch_idx, self.num_classes)
            )
        self.trained = True
        print('info: ending training')
        return 0

    def start_evaluation(self):
        print('info: starting evaluation')
        assert self.trained, 'error: evaluating a non-trained GMM.'
        val_dataloader = self.val_dataloader()
        # test_dataloader = self.test_dataloader()
        val_out = []
        for batch_idx, batch in tqdm.tqdm(enumerate(val_dataloader)):
            val_out.append(self.validation_step(batch, batch_idx))
        res = self.validation_end(val_out)
        print(res)
        print('info: ending evaluation')
        return 0

    def load_model(self, model_path):
        """
        Load GMM model pieces from previous trained ones.
        This method is always called after init on model_manager.py.
        :param model_path:
        :return:
        """
        self.model_path = model_path
        trained_gmm_mask = [isfile(self.model_path / self.gmm_filename.format(class_idx, self.num_classes))
                            for class_idx in range(self.num_classes)]
        self.trained_gmm_indices = [i for i, x in enumerate(trained_gmm_mask) if x]
        for gmm_index in self.trained_gmm_indices:
            self.model.load_gmm(
                gmm_index,
                self.model_path / self.gmm_filename.format(gmm_index, self.num_classes)
            )

        if len(self.trained_gmm_indices) == self.num_classes:
            # if all pieces are found, consider the module fully-trained
            self.trained = True

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
        assert y[0].item() == batch_idx, 'error: batch index is different than a label ({} vs {})'.format(batch_idx,
                                                                                                          y[0].item())
        self.model.fit(x, y)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
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
        x, y = batch['x'], batch['y']
        y_pred = self.forward(x)
        return {'y_pred': y_pred, 'y_target': y}

    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)
        y_pred = None
        y_target = None
        for output in outputs:
            if y_pred is None and y_target is None:
                y_pred = output['y_pred']
                y_target = output['y_target']
            else:
                y_pred = torch.cat((y_pred, output['y_pred']))
                y_target = torch.cat((y_target, output['y_target']))

        np.save('y_pred.npy', y_pred.cpu())
        np.save('y_target.npy', y_target.cpu())

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

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        return [self.optimizer]

    def train_dataloader(self):
        """
        :param trained_gmm_indices: Indices of the batches to avoid loading.
        :return: A lazy iterable of batches
            where every batch should be a dict with keys x and y containing the data and the labels
        """
        class_sampler = ClassSampler(self.num_classes, self.train_dataset.labels, self.train_batch_size)
        trained_gmm_indices = self.trained_gmm_indices

        def batch_generator():
            for batch_idx, indices in enumerate(class_sampler):
                if batch_idx in trained_gmm_indices:
                    print('warning: skipping load of data batch {}, as it was already trained.'.format(batch_idx))
                    yield None
                else:
                    yield self.train_dataset.get_batch(batch_idx, indices)

        return batch_generator

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


class L_AbstractClassifier(ptl.LightningModule):
    """
    Lightning Module template that include standard classification metrics, params and methods.
    Only requires to define self.model and self.optimizer in constructor.
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

        self.metrics = {
            'train_acc': ptl.metrics.Accuracy(),
            'val_acc': ptl.metrics.Accuracy(),
            # 'val_recall': ptl.metrics.Recall(num_classes=num_classes),
            # 'val_precision': ptl.metrics.Precision(num_classes=num_classes),
            # 'val_fbeta': ptl.metrics.Fbeta(num_classes=num_classes),
            # 'val_confmat': ptl.metrics.ConfusionMatrix(num_classes=num_classes),
        }

        # After this constructor should define self.model and self.optimizer

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
        x, y_target = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate metrics
        loss = self.loss(y_pred, y_target)
        self.metrics['train_acc'](y_pred, y_target)
        # log metrics
        self.log('train_loss', loss, prog_bar=True, )
        self.log('train_acc', self.metrics['train_acc'], prog_bar=True, )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y_target = batch['x'], batch['y']
        y_pred = self.forward(x)
        # calculate metrics
        loss = self.loss(y_pred, y_target)
        self.metrics['val_acc'](y_pred, y_target)
        # gather results
        self.log('val_loss', loss, prog_bar=True, )
        self.log('val_acc', self.metrics['train_acc'], prog_bar=True, )
        return loss

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the testing loop

        :param batch:
        :param batch_idx:
        :return:
        """
        x, y_target = batch['x'], batch['y']
        y_pred = self.forward(x)
        return {'y_target': y_target, 'y_pred': y_pred}

    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)
        print(self.model.inter_computations)
        y_pred = None
        y_target = None
        for output in outputs:
            if y_pred is None and y_target is None:
                y_pred = output['y_pred']
                y_target = output['y_target']
            else:
                y_pred = torch.cat((y_pred, output['y_pred']))
                y_target = torch.cat((y_target, output['y_target']))

        np.save('y_pred.npy', y_pred.cpu())
        np.save('y_target.npy', y_target.cpu())

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
        return DataLoader(self.eval_dataset, batch_size=self.batch_size,
                          num_workers=DATA_LOADER_NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=DATA_LOADER_NUM_WORKERS)


class L_WavenetClassifier(L_AbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
        # build model
        self.model = WaveNetClassifier(num_classes)
        summary(self.model, input_size=(1, WAVEFORM_RANDOM_CROP_SEQUENCE_LENGTH), device="cpu")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
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


class L_Conv1DClassifier(L_AbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
        # build model
        self.model = Conv1DClassifier(num_classes)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
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


class L_WavenetTransformerClassifier(L_AbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset)
        # build model
        self.model = WaveNetTransformerClassifier(num_classes)
        summary(self.model, input_size=(1, WAVEFORM_RANDOM_CROP_SEQUENCE_LENGTH), device="cpu")
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


class L_WavenetLSTMClassifier(L_AbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
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


class L_RNNClassifier(L_AbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset, *args, **kwargs)
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


class L_ResNext50(L_AbstractClassifier):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, num_classes, train_dataset, eval_dataset, test_dataset):
        super().__init__(hparams, num_classes, train_dataset, eval_dataset, test_dataset)
        self.model = resnext50_32x4d(num_classes=num_classes)
        self.model.conv1 = Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = x.unsqueeze(1).float()
        return self.model(x)

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
        parser.add_argument('--weight_decay', default=RESNET_V2_WEIGHT_DECAY, type=float)
        parser.add_argument(
            '--distributed_backend',
            type=str,
            default='dp',
            help='supports three options dp, ddp, ddp2'
        )
        return parser
