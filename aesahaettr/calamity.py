import numpy as np
from uvtools import dspec
import tensorflow as tf

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax':tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD':tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


def fit_foregrounds(data, foreground_basis_vectors, weights=None, foreground_coefficients=None, optimizer='RMSprop', **opt_kwargs):
    """A foreground loss function

    Parameters
    ----------
    data: array-like
        Nbls x Nfrequency tensor containing measured data.
    foreground_coefficients: array-like
        Nfg foreground coefficients
    foreground_basis_vectros: array-like
        Nfg x Nbls x Nfrequency tensor of foreground basis vectors.
    optimizer: string
        Name of optimizer. See OPTIMIZERS dictionary
    Returns:

    """
    if weights is None:
        weights = tf.convert_to_tensor(np.ones_like(data.shape, dtype=np.float64))
    else:
        weights = tf.convert_to_tensor(weights)

    if initial_values is None:
        foreground_coefficients = tf.convert_to_tensor(np.random.randn(foreground_basis_vectors.shape[0]))
    else:
        foreground_coefficients = tf.convert_to_tensor(foreground_coefficients)
    foreground_basis_vectors = tf.convert_to_tensor(foreground_basis_vectors)

    loss = lambda: tf.reduce_sum(tf.math.square(tf.math.abs(data - tf.math.cumsum(foreground_coefficients[:, None, None] * foreground_basis_vectors, axis=0) * weights)))

    opt = OPTIMIZERS[optimizer](**opt_kwargs)
    opt.minimize(loss, [foreground_coefficients])
    return foreground_coefficients.numpy()
