# Representation Learning with Contrastive Predictive Coding

### Overview

This is an implementation of:

https://arxiv.org/abs/1807.03748

The above paper experiments with representation learning for audio, vision, natural language, and reinforcement learning. The implementation here is focuses on learning a representation for speech. The training data is from the [LibriSpeech](http://www.openslr.org/12).

### Instructions

1. Get [LibriSpeech data](http://www.openslr.org/12)

Download `dev-clean.tar.gz` and `train-clean-100.tar.gz`

Extract to your home directory:

```
<home>/Data/LibriSpeech/
    dev-clean/
    train-clean-100
```

Note: There are more corrupted datasets available at LibriSpeech. It is probably worthwhile to train a representation with noisier and generally less clean recordings.

2. Create TFRecords files

```bash
python tfrecords.py dev-clean
python tfrecords.py train-clean-100
```

This will add a `tfrecords/` folder to the above `Data` directory.

3. Train the model

```bash
python train.py
```

`-h` to list training options.

4. The encoder weights are saved at the completion of training.

`<cwd>/outputs/genc.h5`

### Requirements

```
tensorflow==2.x
numpy
soundfile
```