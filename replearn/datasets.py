"""
Construct tf.data.Dataset from LibriSpeech TFRecord files.

TFRecord files are built by `tfrecords.py`.
"""
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from pathlib import Path
import glob


class Noise(Sequence):
    """Noisy sequence - confirm autogressive model can't see future."""
    def __init__(self, batch_size=100):
        self.batch_size = batch_size

    def __len__(self):
        return 100

    def __getitem__(self, item):
        num_samps = 20480
        x = np.random.normal(size=(self.batch_size, num_samps)).astype(np.float32)
        y = np.zeros((self.batch_size,)).astype(np.float32)
        return x, y


class SineWaves(Sequence):
    """Generate sine waves with random frequency uniformly: [1kHz, 2kHz]."""
    def __init__(self, batch_size=100):
        self.batch_size = batch_size

    def __len__(self):
        return 100

    def __getitem__(self, item):
        fmin = 1e3
        fmax = 2e3
        num_samps = 20480
        t = np.linspace(0, 1.28, num=num_samps, dtype=np.float32).reshape(1, -1)
        w = (2*np.pi*fmin) + (np.random.random((self.batch_size, 1)) * (fmax-fmin))
        x = np.sin(w * t)  # broadcasts to shape = (batch_size, num_samps)
        y = np.zeros((self.batch_size,)).astype(np.float32)
        return x, y


def decode(example):
    """Parse audio samples from `serialized_example`."""
    feature_description = {'x': tf.io.FixedLenFeature([], tf.string, default_value='')}
    example = tf.io.parse_single_example(example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    x = tf.reshape(x, (-1,))
    y = tf.constant(0, tf.int32)  # auto-regressive task, target always 0

    return x, y


def librispeech_filelist(config, mode):
    if mode == 'train':
        path = Path(config['train_dir'])
    elif mode == 'val':
        path = Path(config['val_dir'])
    else:
        path = None
    return list(map(lambda p: str(p), path.rglob('*.tfr')))


def librispeech(config, mode):
    """Build tf.data.Dataset from LibriSpeech tfrecord files.

    Shuffle when mode == 'train', deterministic order for mode == 'val'

    Parameters
    ----------
    config: dictionary with 'batch_sz' key
    mode: 'train' or 'val'

    Returns
    -------
    tf.data.Dataset with batch size = config['batch_sz']
    """
    filenames = librispeech_filelist(config, mode)
    filenames.sort()
    ds_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(filenames))
    if mode == 'train':
        ds_filenames = ds_filenames.shuffle(len(filenames),
                                            reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(ds_filenames, num_parallel_reads=8)
    ds = ds.map(decode, num_parallel_calls=8)
    if mode == 'train':
        ds = ds.shuffle(1024)
    else:
        ds = ds.prefetch(1024)
    ds = ds.batch(config['batch_sz'])
    ds = ds.prefetch(4)

    return ds
