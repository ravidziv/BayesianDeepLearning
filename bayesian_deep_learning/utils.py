from typing import Union

import numpy as np
import tensorflow as tf


def load_cifar10(normalize: bool = True, batch_size: int = 32) -> Union[tf.data.Dataset, tf.data.Dataset]:
    """Load cifar10 to tf.dataset
    :return: the train and test datasets
    :param batch_size: batch size for train and test
    :param normalize:  if we want to normalize the data
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if normalize:
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-9)
        x_test = (x_test - mean) / (std + 1e-9)
    # todo - Add augmentation
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset
