"""
Model components for Contrastive Predictive Coding.

1. genc_model: encode samples (x_t) to latent (z_t)
2. ar_model: auto-regressive model
3. GenTargets: gerenate positive and negative targets for auto-regressive model
4. ARLoss: auto-regressive loss + prediction accuracy

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Reshape


def genc_model(dim_z):
    """Build CPC encoder model (i.e. g_enc).

    Parameters
    ----------
    dim_z : int
        dimension of latent encoding

    Returns
    -------
    model : keras.Model
        Model that expects time sequence input of shape (N, T) and returns an encoded
        sequence of shape (N, T//256, dim_Z). N is the batch dimension, T is the
        sequence length.
    """
    model = keras.Sequential(name='genc')
    model.add(Lambda(lambda x: K.expand_dims(x, axis=-1)))  # add dim for Conv1D
    model.add(Conv1D(filters=64, kernel_size=8, strides=8, padding='causal',
                     activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv1D(filters=64, kernel_size=4, strides=4, padding='causal',
                     activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv1D(filters=64, kernel_size=4, strides=4, padding='causal',
                     activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv1D(filters=dim_z, kernel_size=4, strides=2, padding='causal',
                     kernel_initializer='he_uniform'))

    return model


def ar_model(t_skip, pred_steps, dim_c, dim_z):
    """Build CPC auto-regressive model (i.e. g_ar).

    Parameters
    ----------
    t_skip : int
        number of c_t steps to crop from beginning of output seqence
    pred_steps : int
        number of future steps to predict
    dim_c : int
        dimension of output context vectors (c_t from fig. 1 of CPC paper)
    dim_z : int
        dimension of input latent vectors (z_t from fig. 1 of CPC paper)

    Returns
    -------
    model : keras.Model
        Model that expects time sequence input of shape (N, t_steps, z_dim) and
        returns encoded sequence of shape (N, t_steps, pred_steps, z_dim).  N is
        the batch dimension.
    """
    model = keras.Sequential(name='ar')
    model.add(GRU(dim_c, return_sequences=True))
    model.add(Lambda(lambda x: K.expand_dims(x, axis=2)))
    model.add(Conv2DTranspose(filters=dim_z, kernel_size=(1, pred_steps)))
    model.add(Cropping2D(((t_skip, pred_steps + 1), (0, 0))))

    return model


class GenTargets(Layer):
    """Generate positive and negative targets for auto-regressive model.

    Notes on negative sampling implementation:
    1) Sampling done with replacement
    2) Possible for occassional postive examples included with negatives, with
       probability ~= 1/target_len. Sampling code was easier to write this way.
    3) Negative examples not selected across batch, only within example. This
       approach had consistently best performance in papers.
    4) For easy coding, negative indices same for all batch examples.
    TODO - revisit 2 and 4, and evaluate performance
    """
    def __init__(self, t_skip, pred_steps, num_neg, **kwargs):
        """
        Parameters
        ----------
        t_skip : int
            number of c_t steps to crop from beginning of output seqence
        pred_steps : int
            number of future steps to predict
        num_neg : int
            number of negative samples
        """
        super(GenTargets, self).__init__(**kwargs)
        self.t_skip = t_skip
        self.pred_steps = pred_steps
        self.num_neg = num_neg
        self.batch_sz = None    # set in .build()
        self.t = None
        self.dim_z = None
        self.target_len = None

    def build(self, input_shape):
        self.batch_sz, self.t, self.dim_z = input_shape
        # target_len is smaller than total input time steps (self.t)
        # 1. skip `t_start` steps (GRU "warm-up" time + skip conv1d zero padding)
        # 2. can't predict beyond end of seqence: subtract pred_steps
        # 3. prediction offset by 1 (predict future only): subtract 1
        self.target_len = self.t - self.t_skip - self.pred_steps - 1

    def call(self, inputs, **kwargs):
        inputs = K.stop_gradient(inputs)
        # sample negative targets
        num_samples = self.num_neg * self.target_len * self.pred_steps
        neg_indices = K.random_uniform((num_samples,), self.t_skip, self.t - 1, 'int32')
        negatives = tf.gather(inputs, neg_indices, axis=1)
        negatives = Reshape((self.num_neg, self.target_len,
                             self.pred_steps, self.dim_z))(negatives)
        # build positive targets
        inputs = K.expand_dims(inputs, axis=2)  # add pred_step axis
        # inputs.shape() = (batch_sz, input_seq_len, 1, z_dim)
        positives_list = []
        for i in range(self.pred_steps):
            start = self.t_skip + i + 1  # +1 because predicting future only
            positives_list.append(inputs[:, start:(start+self.target_len), ...])
        positives = K.concatenate(positives_list, axis=2)
        positives = K.expand_dims(positives, axis=1)  # add axis for pos+neg concat
        # positives.shape() = (batch_sz, 1, target_len, pred_steps, z_dim)

        return K.concatenate([positives, negatives], axis=1)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                't_skip': self.t_skip,
                'pred_steps': self.pred_steps,
                'num_neg': self.num_neg,
                }

    def compute_output_shape(self, input_shape):
        batch_sz, t, dim_z = input_shape
        target_len = t - self.t_skip - self.pred_steps - 1
        target_shape = (batch_sz, self.num_neg + 1,
                        target_len, self.pred_steps, dim_z)

        return target_shape


class ARLoss(Layer):
    """Contrastive loss function + prediction accuracy.

    Equation 4 from CPC paper.
    """
    def call(self, inputs, **kwargs):
        # TODO - optimized back-prop with K.categorical_cross_entropy()?
        z, z_hat = inputs
        # z.shape() = (B, neg+1, T, pred_steps, dim_z)
        z_hat = K.expand_dims(z_hat, axis=1)  # add pos/neg example axis
        # z_pred.shape() = (B, 1, T, pred_steps, dim_z)
        logits = K.sum(z * z_hat, axis=-1)  # dot product
        # logits.shape() = (B, neg+1, T, pred_steps)
        log_ll = logits[:, 0, ...] - tf.math.reduce_logsumexp(logits, axis=1)
        # log_ll.shape() = (B, T, pred_steps)
        loss = -K.mean(log_ll, axis=[1, 2])
        # calculate prediction accuracy
        acc = K.cast(K.equal(K.argmax(logits, axis=1), 0), 'float32')
        acc = K.mean(acc, axis=[0, 1])

        return loss, acc

    def compute_output_shape(self, input_shape):
        return input_shape,
