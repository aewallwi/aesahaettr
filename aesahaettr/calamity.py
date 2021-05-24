import numpy as np
from uvtools import dspec
import tensorflow as tf

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax':tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD':tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


def calibrate_data(data, foreground_basis_vectors, weights=None, foreground_coefficients=None, optimizer='RMSprop', tol=1e-6, **opt_kwargs):
    """A foreground loss function

    Parameters
    ----------
    data: array-like
        (Nbls x Nfrequency) length 1d tensor containing data.
    foreground_coefficients: array-like
        Nfg foreground coefficients
    foreground_basis_vectros: array-like
        (Nbls x Nfrequency) x Nfg 2d tensor of foreground basis vectors.
    optimizer: string
        Name of optimizer. See OPTIMIZERS dictionary
    Returns:

    """
