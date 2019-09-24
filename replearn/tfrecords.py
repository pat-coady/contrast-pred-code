#! /usr/bin/env python
"""
Load LibriSpeech sound clips, break into shorter chunks and write to
TFRecords files (each with `shard_size` examples).

Download LibriSpeech files here: http://www.openslr.org/12

Expects LibriSpeech *.flac files to be untarred to ~/Data/LibriSpeec/...
"""
import numpy as np
import tensorflow as tf
import soundfile as sf
import random
from pathlib import Path
import argparse


# from: https://www.tensorflow.org/alpha/tutorials/load_data/tf_records
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_all_filenames(data_split):
    search_path = (Path.home() / 'Data' / 'LibriSpeech' / data_split)
    filenames = search_path.rglob('*.flac')

    return list(map(lambda fn: str(fn), filenames))


class TFRWriter(object):
    """Manage shard counts, filenames, and open/close tfr files."""
    def __init__(self, data_split, shard_size=1024):
        self.shard_size = shard_size
        self.path = (Path.home() / 'Data' / 'LibriSpeech' / 'tfrecords' / data_split)
        self.path.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.shard_num = 0
        self.writer = tf.io.TFRecordWriter(self._tfr_name())

    def write(self, example):
        self.writer.write(example)
        self.count += 1
        if self.count % self.shard_size == 0:
            self.writer.close()
            self.shard_num += 1
            self.writer = tf.io.TFRecordWriter(self._tfr_name())

    def close(self):
        self.writer.close()

    def _tfr_name(self):
        return str(self.path / '{:04d}.tfr'.format(self.shard_num))


def serialized_example(x):
    feature = {'x': _bytes_feature(tf.io.serialize_tensor(x))}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def main(data_split):
    shard_size = 1024
    samples_per_clip = 20480
    filenames = get_all_filenames(data_split)
    random.seed(0)
    random.shuffle(filenames)
    writer = TFRWriter(data_split, shard_size)
    for filename in filenames:
        samples, sample_rate = sf.read(filename)
        # center and normalize amplitude
        samples = samples.astype(np.float32)
        samples = samples - samples.mean()
        max_amplitude = np.max([-samples.min(), samples.max()])
        samples = samples / max_amplitude
        if sample_rate != 16000:
            continue
        num_samples = samples.shape[0]
        num_chunks = num_samples // samples_per_clip
        # start at random spot in each clip:
        rand_start = random.randrange(num_samples % samples_per_clip + 1)
        for i in range(num_chunks):
            start = i * samples_per_clip + rand_start
            end = start + samples_per_clip
            x = samples[start:end]
            example = serialized_example(x)
            writer.write(example)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build TFRecord files from LibriSpeech *.flac files.'
    )
    parser.add_argument('data_split', type=str,
                        help='Name of data split: "dev-clean" or "train-clean-100".')
    main(parser.parse_args().data_split)
