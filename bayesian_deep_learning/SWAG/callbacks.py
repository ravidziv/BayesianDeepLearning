from typing import Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from typing_extensions import Literal

from bayesian_deep_learning.SWAG.swag_utils import sample, average_weights, initial_swag_weights


class SWAGCallback(Callback):
    """Stochastic Weight Averaging.
    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407
    # Arguments
        start_epoch:   integer, epoch when swa should start.
        lr_schedule:   string, type of learning rate schedule.
        swag_lr:        float, learning rate for swa.
        swag_lr2:       float, upper bound of cyclic learning rate.
        swag_freq:      integer, length of learning rate cycle.
        batch_size     integer, batch size (for batch norm with generator)
        verbose:       integer, verbosity mode, 0 or 1.
    """

    def __init__(
            self,
            start_epoch,
            swa_freq=1,
            batch_size=None,
            var_clamp: float = 1e-30,
            var_top_clamp: float = 1000.,
            verbose=0,
    ):

        super(SWAGCallback, self).__init__()
        self.start_epoch = start_epoch - 1
        self.swa_freq = swa_freq
        self.batch_size = batch_size
        self.verbose = verbose
        self.var_clamp = var_clamp
        self.var_top_clamp = var_top_clamp
        tf.debugging.assert_greater(start_epoch, 2, message='"swa_start" attribute cannot be lower than 2.')

    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs")

        tf.debugging.assert_greater_equal(self.epochs - 1, self.start_epoch, message='"swa_start" attribute must be '
                                                                                     'lower than "epochs".')
        self._check_batch_norm()
        condition = self.has_batch_norm and self.batch_size is None
        tf.debugging.assert_equal(condition, False, message='"batch_size" needs to be set for models with batch '
                                                            'normalization layers.')

    def on_epoch_begin(self, epoch, logs=None):
        self._scheduler(epoch)
        if self.is_swa_start_epoch:
            self.swa_weights = initial_swag_weights(model_weights=self.model.swag_model.get_weights())
        if self.is_batch_norm_epoch:
            self._reset_batch_norm()

    def on_batch_begin(self, batch, logs=None):
        if self.is_batch_norm_epoch:
            batch_size = self.batch_size
            momentum = batch_size / (batch * batch_size + batch_size)

            for layer in self.batch_norm_layers:
                layer.momentum = momentum

    def on_epoch_end(self, epoch, logs=None):
        if self.is_swa_epoch and not self.is_batch_norm_epoch:
            self.swa_weights = average_weights(epoch=epoch, start_epoch=self.start_epoch,
                                               swa_weights=self.swa_weights,
                                               model_weights=self.model.base_model.get_weights(),
                                               swa_freq=self.swa_freq)
            self._set_swa_weights()

    def on_train_end(self, logs=None):
        if not self.has_batch_norm:
            self._set_swa_weights()
        else:
            self._restore_batch_norm()

    def _scheduler(self, epoch):
        swa_epoch = epoch - self.start_epoch
        self.is_swa_epoch = (
                epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
        )
        self.is_swa_start_epoch = epoch == self.start_epoch
        self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm

    def _set_swa_weights(self, scale=0.0, block=False):
        samples = sample(scale=scale, block=block, swag_weights=self.swa_weights, var_clamp=self.var_clamp,
                         var_top_clamp=self.var_top_clamp)
        self.model.swag_model.set_weights(samples)

    def _check_batch_norm(self):
        self.batch_norm_momentums = []
        self.batch_norm_layers = []
        self.has_batch_norm = False
        self.running_bn_epoch = False
        for layer in self.model.swag_model.layers:
            if issubclass(layer.__class__, BatchNormalization):
                self.has_batch_norm = True
                self.batch_norm_momentums.append(layer.momentum)
                self.batch_norm_layers.append(layer)

    def _reset_batch_norm(self):
        for layer in self.batch_norm_layers:
            # to get properly initialized moving mean and moving variance weights
            # we initialize a new batch norm layer from the config of the existing
            # layer, build that layer, retrieve its reinitialized moving mean and
            # moving var weights and then delete the layer
            bn_config = layer.get_config()
            new_batch_norm = BatchNormalization(**bn_config)
            new_batch_norm.build(layer.input_shape)
            new_moving_mean, new_moving_var = new_batch_norm.get_weights()[-2:]
            # get rid of the new_batch_norm layer
            del new_batch_norm
            # get the trained gamma and beta from the current batch norm layer
            trained_weights = layer.get_weights()
            new_weights = []
            # get gamma if exists
            if bn_config["scale"]:
                new_weights.append(trained_weights.pop(0))
            # get beta if exists
            if bn_config["center"]:
                new_weights.append(trained_weights.pop(0))
            new_weights += [new_moving_mean, new_moving_var]
            # set weights to trained gamma and beta, reinitialized mean and variance
            layer.set_weights(new_weights)

    def _restore_batch_norm(self):
        for layer, momentum in zip(
                self.batch_norm_layers, self.batch_norm_momentums
        ):
            layer.momentum = momentum


class SWAGRScheduler(Callback):
    def __init__(self, lr_schedule: Literal["manual", "constant", "cyclic"], swag_lr2: Union[str, float],
                 swag_lr: Union[str, float], start_epoch: int, swag_freq: int):
        """learning rate scheduler for SWAGCallback
        :param lr_schedule: which scheduler to choose
        :param swag_lr2: max learning rate for SWAG
        :param swag_lr: lower bound learning right for SWAG
        :param start_epoch: which epoch start SWAG
        :param swag_freq: the interval to update the parameters
        """
        super().__init__()
        self.lr_schedule = lr_schedule
        # if no user determined upper bound, make one based off of the lower bound
        self.swa_lr2 = swag_lr2 if swag_lr2 is not None else 10 * swag_lr
        self.swa_lr = swag_lr
        self.swa_freq = swag_freq
        self.start_epoch = start_epoch - 1
        condition = self.swa_lr == "auto" and self.swa_lr2 == 'auto'
        tf.debugging.assert_equal(condition, False,
                                  message='"swag_lr2" cannot be manually set if "swag_lr" is automatic.')
        condition = "cyclic" == self.lr_schedule and self.swa_lr != "auto" and self.swa_lr2 != "auto" \
                    and self.swa_lr > self.swa_lr2
        tf.debugging.assert_equal(condition, False, message='"swag_lr" must be lower than "swag_lr2".')

    def on_train_begin(self, logs=None):
        self.init_lr = K.eval(self.model.optimizer.lr)
        if self.lr_schedule == "cyclic" and self.swa_lr2 == "auto":
            self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr) * 0.25
            # automatic swag_lr
        if self.swa_lr == "auto":
            self.swa_lr = 0.1 * self.init_lr
        tf.debugging.assert_greater(self.init_lr, self.swa_lr, message='"swag_lr" must be lower than rate set.')
        if self.lr_schedule == "cyclic" and self.swa_lr2 == "auto":
            self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr) * 0.25

    def on_epoch_begin(self, epoch, logs=None):
        # constant schedule is updated epoch-wise
        self.current_epoch = epoch
        # steps are mini-batches per epoch, equal to training_samples / batch_size
        steps = self.params.get("steps")
        if self.lr_schedule == "constant":
            update_lr(epoch=epoch, batch=None, start_epoch=self.start_epoch,
                      swa_lr=self.swa_lr, steps=steps, swa_freq=self.swa_freq, swa_lr2=self.swa_lr2,
                      init_lr=self.init_lr,
                      optimizer_lr=self.model.optimizer.lr, lr_schedule=self.lr_schedule)

    def on_batch_begin(self, batch, logs=None):
        # update lr each batch for cyclic lr schedule
        # steps are mini-batches per epoch, equal to training_samples / batch_size
        steps = self.params.get("steps")
        if self.lr_schedule == "cyclic":
            update_lr(epoch=self.current_epoch, batch=batch, start_epoch=self.start_epoch,
                      swa_lr=self.swa_lr, steps=steps, swa_freq=self.swa_freq, swa_lr2=self.swa_lr2,
                      init_lr=self.init_lr, optimizer_lr=self.model.optimizer.lr, lr_schedule=self.lr_schedule)
        tf.summary.scalar('learning rate', data=self.model.optimizer.lr, step=batch * (self.current_epoch + 1))


def update_lr(epoch: int, start_epoch: int, swa_lr: float, steps: int, swa_freq: int, swa_lr2: float, init_lr: float,
              batch: int = None, optimizer_lr: float = None,
              lr_schedule: str = "constant") -> None:
    """

    :rtype: None
    :param epoch: current epoch
    :param start_epoch: the first epoch to update SWAG
    :param swa_lr: min learning rate for SWAG
    :param steps: steps till now
    :param swa_freq: interval for update SWAG
    :param swa_lr2: max learning rate for SWAG
    :param init_lr: the initial lr for the optimizer
    :param batch: current batch number
    :param optimizer_lr: current lr
    :param lr_schedule: scheduler for the lr
    """
    # if is_batch_norm_epoch:
    #    lr = 0
    #    K.set_value(optimizer_lr, lr)
    if lr_schedule == "constant":
        lr = _constant_schedule(epoch=epoch, start_epoch=start_epoch, swa_lr=swa_lr, init_lr=init_lr)
        K.set_value(optimizer_lr, lr)
    elif lr_schedule == "cyclic":
        lr = _cyclic_schedule(epoch=epoch, steps=steps, start_epoch=start_epoch, swa_freq=swa_freq, swa_lr2=swa_lr2,
                              swa_lr=swa_lr, batch=batch, init_lr=init_lr)
        K.set_value(optimizer_lr, lr)


def _constant_schedule(epoch: int, start_epoch: int, swa_lr: float, init_lr: float) -> float:
    """
    Calculate the updated learning rarte which change by constant factor each step.
    :return: updated learning rate
    :rtype: float
    :param epoch: current epoch
    :param start_epoch:  start epoch for SWAG
    :param swa_lr: min lr for SWAG
    :param init_lr: initial optimizer lr
    """
    t = epoch / start_epoch
    lr_ratio = swa_lr / init_lr
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return init_lr * factor


def _cyclic_schedule(epoch: int, steps: int, start_epoch: int, swa_freq: int, swa_lr2: float, swa_lr: float, batch: int,
                     init_lr: float) \
        -> float:
    """Designed after Section 3.1 of Averaging Weights Leads to
    Wider Optima and Better Generalization(https://arxiv.org/abs/1803.05407)
    :param steps:
    :param swa_freq:
    :param swa_lr2:
    :param batch:
    :return:
    :rtype: float
    :param epoch: current epoch
    :param start_epoch:  start epoch for SWAG
    :param swa_lr: min lr for SWAG
    :param init_lr: initial optimizer lr
    :return:  the updated learning rate
    """
    # occasionally steps parameter will not be set. We then calculate it ourselves
    # if steps == None:
    #    steps = params["samples"] // self.params["batch_size"]
    swa_epoch = (epoch - start_epoch) % swa_freq
    cycle_length = swa_freq * steps
    # batch 0 indexed, so need to add 1
    i = (swa_epoch * steps) + (batch + 1)
    if epoch >= start_epoch:
        t = (((i - 1) % cycle_length) + 1) / cycle_length
        return (1 - t) * swa_lr2 + t * swa_lr
    else:
        return _constant_schedule(epoch=epoch, start_epoch=start_epoch, swa_lr=swa_lr, init_lr=init_lr)
