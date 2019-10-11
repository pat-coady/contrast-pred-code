#! /usr/bin/env python
"""
Contrastive Predictive Coding: Audio Representation Learning

https://arxiv.org/abs/1807.03748
"""
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

import argparse
import shutil
import json
from pathlib import Path

from models import genc_model, ar_model
from models import GenTargets, ARLoss
from datasets import librispeech
from callbacks import build_callbacks


def add_logpaths(config):
    """Add paths for checkpoints and TensorBoard data to the config dictionary."""
    config['ckpt_path'] = str(Path.cwd() / 'outputs' / 'checkpoints')
    config['tblog_path'] = str(Path.cwd() / 'logs')
    Path(config['ckpt_path']).mkdir(parents=True, exist_ok=True)


def save_state(config):
    """Save configuration of training run: all *.py files and config dictionary."""
    filenames = Path.cwd().glob('*.py')
    for filename in filenames:
        shutil.copy(str(filename), config['ckpt_path'])
    with (Path(config['ckpt_path']) / 'config.json').open('w') as f:
        json.dump(config, f)


def init_azure_logging(config):
    """Get Azure run logger and log all configuration settings."""
    from azureml.core.run import Run
    run_logger = Run.get_context()
    for k in config:
        run_logger.log(k, config[k])

    return run_logger


def build_model(config):
    """Build CPC model and prediction accuracy model.

    Parameters
    ----------
    config : dict
        configuration dictionary, mostly built from command line argparse

    Returns
    -------
    model : keras.Model
        Contrastive Predictive Coding model. Model input: batch of audio samples, each of
        length 20480. Model output: autogressive loss (Eq. 4 in CPC paper).
    acc_model : keras.Model
        Batch prediction accuracy for each of future prediction steps. For example, if
        CPC is configure to predict 12 future steps, acc_model will output a vector of
        shape=(12,) with the accuracy of correctly predicting the positive example vs.
        the negative examples.
    """
    samples_per_clip = 20480
    inputs = Input(shape=(samples_per_clip,), name='inputs')
    genc = genc_model(dim_z=config['dim_z'])  # encoder
    z = genc(inputs)
    ar = ar_model(**config)  # auto-regressive model
    z_hat = ar(z)
    gen_targets = GenTargets(t_skip=config['t_skip'],
                             pred_steps=config['pred_steps'],
                             num_neg=config['num_neg'],
                             name='gen_targets')
    z_targ = gen_targets(z)  # shift and crop auto-regressive targets, add negative samples
    loss_fn = ARLoss(name='ar_loss')       # auto-regressive loss
    loss, acc = loss_fn([z_targ, z_hat])  # loss calculated in model, auto-regressive task
    model = Model(inputs=inputs, outputs=loss)
    # acc_model: accuracy metric for each future time step. shape=(pred_steps,)
    acc_model = Model(inputs=inputs, outputs=acc)

    return model, acc_model


def get_optimizer(config):
    if config['optimizer'] == 'adam':
        return tf.optimizers.Adam(config['lr'])
    if config['optimizer'] == 'sgd_mom':
        return tf.optimizers.SGD(config['lr'], momentum=0.9, nesterov=True)


def main(config):
    run_logger = None
    if config['azure_ml']:
        run_logger = init_azure_logging(config)
    ds_train = librispeech(config, 'train')
    ds_val = librispeech(config, 'val')
    model, acc_model = build_model(config)
    optimizer = get_optimizer(config)
    # Note: Model is auto-regressive and returns a loss. So, the training target is 0
    # with an objective of minimizing the mean average error.
    model.compile(loss='mae', optimizer=optimizer)
    add_logpaths(config)
    save_state(config)
    callbacks = build_callbacks(config, run_logger, acc_model, ds_val)
    model.fit(x=ds_train, validation_data=ds_val, epochs=config['epochs'],
              validation_steps=100, callbacks=callbacks, verbose=config['verbose'])
    # save encoding layer weights:
    model.get_layer('genc').save_weights(str(Path.cwd() / 'outputs' / 'genc.h5'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '-lr', '--lr', type=float,
        help='Learning rate (default=0.001)',
        default=0.001)
    parser.add_argument(
        '-e', '--epochs', type=int,
        help='Number of epochs (default=25)',
        default=25)
    parser.add_argument(
        '-b', '--batch_sz', type=int,
        help='Batch size (default=64)',
        default=64)
    parser.add_argument(
        '-dc', '--dim_c', type=int,
        help='Dimension of context vector (default=40).',
        default=40)
    parser.add_argument(
        '-dz', '--dim_z', type=int,
        help='Dimension of encoding vector (default=40).',
        default=40)
    parser.add_argument(
        '--train_dir', type=str,
        help='Path to folder containing training *.tfr files.',
        default=str(Path.home() / 'Data' / 'LibriSpeech' / 'tfrecords' / 'train-clean-100'))
    parser.add_argument(
        '--val_dir', type=str,
        help='Path to folder containing validation *.tfr files.',
        default=str(Path.home() / 'Data' / 'LibriSpeech' / 'tfrecords' / 'dev-clean'))
    parser.add_argument(
        '--optimizer', type=str,
        help='Choose optimizer: "adam" or "sgd_mom". (default=adam)',
        default='adam')
    parser.add_argument(
        '-ts', '--t_skip', type=int,
        help='Number context vectors to skip at beginning: "RNN warmup". (default=0)',
        default=0)
    parser.add_argument(
        '-ps', '--pred_steps', type=int,
        help='Number of future encoding steps to predict. (default=10)',
        default=10)
    parser.add_argument(
        '--num_neg', type=int,
        help='Number of negative samples. (default=10)',
        default=10)
    parser.add_argument(
        '--azure_ml', action='store_true',
        help='Enable Azure ML logging.')
    parser.set_defaults(azure_ml=False)
    parser.add_argument(
        '-v', '--verbose', type=int,
        help='Verbose level for Keras Model.fit(). (default=2)',
        default=2
    )

    main(vars(parser.parse_args()))
