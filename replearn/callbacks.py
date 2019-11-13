"""
Custom Keras callbacks for logging and checkpointing.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import csv


class AzureLoggingCallback(tf.keras.callbacks.Callback):
    """Logging for Azure ML framework."""
    def __init__(self, run_logger):
        super(AzureLoggingCallback, self).__init__()
        self.run_logger = run_logger

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            self.run_logger.log(k, logs[k])


class AccCallback(tf.keras.callbacks.Callback):
    """Measure prediction accuracy: positive examples correctly predicted."""
    def __init__(self, acc_model, acc_data, run_logger):
        super(AccCallback, self).__init__()
        self.acc_model = acc_model
        self.acc_data = acc_data
        self.history = []
        path = Path.cwd() / 'outputs'
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / 'accuracy.csv'
        f = self.path.open('w')
        f.close()  # create empty accuracy log file
        self.run_logger = run_logger

    def on_epoch_end(self, epoch, logs=None):
        self._calc_accs()

    def on_train_begin(self, logs=None):
        self._calc_accs()

    def _calc_accs(self):
        accs = []
        for x in self.acc_data:
            accs.append(self.acc_model(x).numpy()[np.newaxis, :])
        accs.pop()  # remove stats from last batch - probably a partial batch
        a = np.mean(np.concatenate(accs), axis=0)
        print('Step accuracies:')
        print(a)
        with self.path.open('a') as f:
            writer = csv.writer(f)
            writer.writerow(a)
        if self.run_logger is not None:
            for i in [0, 1, 2, 5, 9]:
                if i < a.shape[0]:
                    self.run_logger.log('step{}_acc'.format(i), a[i])


def build_callbacks(config, run_logger, acc_model, val_data):
    """Build list of callbacks to monitor training."""
    callbacks = []
    cb = keras.callbacks.ModelCheckpoint(config['ckpt_path'] + '/cp-{epoch:04d}.ckpt',
                                         save_weights_only=True)
    callbacks.append(cb)
    cb = keras.callbacks.TensorBoard(config['tblog_path'],
                                     histogram_freq=1,
                                     update_freq=500)
    callbacks.append(cb)
    cb = AccCallback(acc_model, val_data, run_logger)
    callbacks.append(cb)

    if config['azure_ml']:
        cb = AzureLoggingCallback(run_logger)
        callbacks.append(cb)

    return callbacks
