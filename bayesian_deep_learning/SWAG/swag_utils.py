"""Utils for SWAG keras"""
import collections
from typing import List

import numpy as np
import tensorflow as tf

GAUSSIAN = collections.namedtuple('gaussian', ['mean', 'var'])


def sample(scale: float = 1.0, block: bool = False, swag_weights: List[GAUSSIAN] = None, var_clamp: float = 1e-30,
           var_top_clamp: float = 1e4) -> List[tf.Tensor]:
    """Sample from the swa weights
    :param scale: the shift in the gaussian
    :param block: calculate full bloc wise gaussian or not
    :param swag_weights: the current SWAG weights
    :param var_clamp: min value for the gaussian variance
    :param var_top_clamp: max value for the gaussian variance
    :return: the new sampled weights
    :rtype: List[tf.Tensor]
    """
    samples = None
    if not block:
        samples = sample_full_rank(scale=scale, swag_weights=swag_weights,
                                   var_clamp=var_clamp, var_top_clamp=var_top_clamp)
    else:
        pass
        # todo add implementation of block wise sample
        # samples = sample_block_wise(scale, cov)
    return samples


def calc_full_rank(mean: float, sq_mean: float, scale_sqrt: float, var_clamp: float, var_top_clamp: float) -> float:
    """
    Calculate full rank sample from given mean and variance
    :rtype: float
    :param mean: the mean of the gaussian
    :param sq_mean:  the variance of the gaussian
    :param scale_sqrt:  the shift of the mean
    :param var_clamp: min value of variance
    :param var_top_clamp: max value of variance
    :return: the sampled value
    """
    # draw diagonal variance current_sample
    var = tf.clip_by_value(sq_mean - mean ** 2, var_clamp, var_top_clamp)
    var_sample = tf.cast(tf.sqrt(var), dtype=tf.float32) * tf.random.normal(shape=var.shape)
    # update current_sample with mean and scale
    current_sample = mean + scale_sqrt * var_sample
    return current_sample


def sample_full_rank(scale: float, swag_weights: List[GAUSSIAN], var_clamp: float, var_top_clamp: float) \
        -> List[tf.Tensor]:
    """
    Sample the weights from the gassing with full rank
    :rtype: List[tf.Tensor]
    :param scale: scaling of the mean of the gaussian
    :param swag_weights: the current gaussian weights
    :param var_clamp: min variance
    :param var_top_clamp: max variance 
    :return: List of new sampled weights 
    """
    scale_sqrt = scale ** 0.5
    samples = []
    # Go over all the weights and sample
    # todo Do it in parallel
    for w in swag_weights:
        mean = w.mean
        sq = w.var
        current_sample = calc_full_rank(mean.reshape(-1), sq.reshape(-1), scale_sqrt, var_clamp, var_top_clamp)
        samples.append(tf.reshape(current_sample, mean.shape))
    return samples


def average_weights(epoch: int, start_epoch: int, swa_weights: list, model_weights: tf.Tensor, swa_freq: int) \
        -> List[GAUSSIAN]:
    """Calculate a list of Gaussian for the weights based on their average
    :rtype: List[GAUSSIAN]
    :param epoch: current epoch
    :param start_epoch:  start epoch of SWAG
    :param swa_weights: list with all the previous SWAG weights
    :param model_weights: the current weights of the model
    :param swa_freq: The interval for SWAG calculations
    :return:
    """
    swa = []
    n_models = (epoch - start_epoch)
    for swa_w, w_mean in zip(swa_weights, model_weights):
        swa_w_mean = tf.experimental.numpy.copy(swa_w.mean)
        swa_w_sq = tf.experimental.numpy.copy(swa_w.var)
        mean = (swa_w_mean * (n_models / swa_freq) + w_mean) / (n_models + 1)
        sq_mean = (swa_w_sq * (n_models / swa_freq) + w_mean ** 2) / (n_models + 1)
        new_swa_w = GAUSSIAN(mean=mean, var=sq_mean)
        swa.append(new_swa_w)
    return swa


def initial_swag_weights(model_weights: np.array) -> List[GAUSSIAN]:
    """Crete for each weights a gaussian with the correct weights as mean and variance 0
    :param model_weights:
    :return: list of the Gaussian for all the weights
    :rtype: list[Gaussian]
    """
    swa_weights = []
    for w in model_weights:
        current_swa = GAUSSIAN(mean=w, var=0)
        swa_weights.append(current_swa)
    return swa_weights
