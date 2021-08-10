import argparse
import os
from datetime import datetime

import tensorflow as tf

from bayesian_deep_learning.SWAG.callbacks import SWAGCallback, SWAGRScheduler
from bayesian_deep_learning.SWAG.models import SWAG_model
from bayesian_deep_learning.utils import load_cifar10


def run(args):
    # make datasets
    train_dataset, test_dataset = load_cifar10(batch_size=args.batch_size, normalize=True)
    # build model
    model = tf.keras.applications.vgg16.VGG16(
        include_top=True, weights=None, input_tensor=None, pooling=None, classes=10,
        classifier_activation='linear', input_shape=(32, 32, 3)
    )
    # Wrap the model with SWAGCallback
    model = SWAG_model(model, model)
    loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    sgd = tf.keras.optimizers.SGD(lr=args.learning_rate, decay=args.lr_decay, momentum=args.momentum, nesterov=False)
    # sgd = tf.keras.optimizers.Adam(1e-4)
    regular_loss_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=True, name='regular_loss')
    model.compile(loss=loss,
                  optimizer=sgd, metrics=[regular_loss_metric, tf.keras.metrics.CategoricalAccuracy('Accuracy')])
    # define swag callback
    logdir = os.path.join(args.logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    swag = SWAGCallback(start_epoch=args.swag_start_epoch)
    swag_sch = SWAGRScheduler(start_epoch=args.swag_start_epoch, lr_schedule=args.lr_schedule,
                              swag_lr2=args.swag_lr2, swag_lr=args.swag_lr,
                              swag_freq=args.swag_freq)
    model.run_eagerly  = True
    callbacks = [tensorboard_callback, swag, swag_sch]
    model.fit(train_dataset, validation_data=test_dataset, steps_per_epoch=args.steps_per_epoch,
              validation_steps=args.validation_steps,
              epochs=args.epochs,
              verbose=1, callbacks=callbacks)
    model.evaluate(test_dataset, steps=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main for self-supervised learning")
    parser.add_argument('--task', type=str, default='classification',
                        help="Specify the task; either pretrain, classification or segmentation")
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.015, help='learning rate for the train')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the train')
    parser.add_argument('--lr_decay', type=float, default=1e-6, help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--swag_start_epoch', type=int, default=10,
                        help='from which epoch start calculate SWAGCallback')
    parser.add_argument('--swag_freq', type=int, default=1, help='the interval for calculating swag')
    parser.add_argument('--swag_lr', type=float, default=0.01, help='min lr for SWAGCallback')
    parser.add_argument('--swag_lr2', type=str, default='auto', help='max lr for SWAGCallback - can be auto')
    parser.add_argument('--lr_schedule', type=str, default='constant', help='how to schedule the lr')
    parser.add_argument('--steps_per_epoch', type=int, default=300, help='steps per epoch for training')
    parser.add_argument('--validation_steps', type=int, default=30, help='steps for validation')
    parser.add_argument('--logdir', type=str, default='/data/BDL/logs', help='where to store the models')
    args = parser.parse_args()
    run(args)
