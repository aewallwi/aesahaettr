import numpy as np
from uvtools import dspec
import tensorflow as tf

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax':tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD':tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


def fit_foregrounds(data, foreground_basis_vectors, weights=None, foreground_coefficients=None, optimizer='RMSprop', tol=1e-6, **opt_kwargs):
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
    if weights is None:
        weights = tf.convert_to_tensor(np.ones(data.shape, dtype=np.float64))
    else:
        weights = tf.convert_to_tensor(weights)
    scale_factor = np.sqrt(np.mean(np.abs(data) ** 2.))
    if foreground_coefficients is None:
        foreground_coefficients_real = tf.Variable(tf.convert_to_tensor(np.random.randn(foreground_basis_vectors.shape[0])))
        foreground_coefficients_imag = tf.Variable(tf.convert_to_tensor(np.random.randn(foreground_basis_vectors.shape[0])))
    else:
        foreground_coefficients_real = tf.Variable(tf.convert_to_tensor(foreground_coefficients.real / scale_factor))
        foreground_coefficients_imag = tf.Variable(tf.convert_to_tensor(foreground_coefficients.imag / scale_factor))
    foreground_basis_vectors = tf.convert_to_tensor(foreground_basis_vectors)
    data_real = tf.convert_to_tensor(data.real / scale_factor)
    data_imag = tf.convert_to_tensor(data.imag / scale_factor)

    loss = lambda: tf.reduce_sum(tf.math.square(tf.math.abs(data_real - tf.math.reduce_sum(foreground_basis_vectors * foreground_coefficients_real, axis=1) * weights))) \
                   + tf.reduce_sum(tf.math.square(tf.math.abs(data_imag - tf.math.reduce_sum(foreground_basis_vectors * foreground_coefficients_imag, axis=1) * weights)))

    opt = OPTIMIZERS[optimizer](**opt_kwargs)

    foreground_coefficients_real_last = foreground_coefficients_real.numpy()
    foreground_coefficients_imag_last = foreground_coefficients_imag.numpy()
    opt.minimize(loss, [foreground_coefficients_real, foreground_coefficients_imag])

    delta_real = np.abs(foreground_coefficients_real.numpy() - foreground_coefficients_real_last).max()
    delta_imag = np.abs(foreground_coefficients_imag.numpy() - foreground_coefficients_imag_last).max()

    while delta_real >= tol and delta_imag >= tol:
        opt.minimize(loss, [foreground_coefficients_real, foreground_coefficients_imag])
        delta_real = np.abs(foreground_coefficients_real.numpy() / foreground_coefficients_real_last - 1.).max()
        delta_imag = np.abs(foreground_coefficients_imag.numpy() / foreground_coefficients_imag_last - 1.).max()
        foreground_coefficients_real_last = foreground_coefficients_real.numpy()
        foreground_coefficients_imag_last = foreground_coefficients_imag.numpy()

    return (foreground_coefficients_real.value().numpy() + 1j * foreground_coefficients_imag.value().numpy()) * scale_factor
