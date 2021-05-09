# tools for computing ivsibility covariances.

import numpy as np
from . import visibilities

def cov_mat_simple(uvd=None, antenna_chromaticity=0.0, bl_cutoff_buffer=np.inf, order_by_bl_length=False,
                   return_bl_lens_freqs=False, **array_config_kwargs):
    """
    A covariance matrix that is both simple and flexible enough to describe
    the covariances within the wedge and simple antennas.

    Parameters
    ----------
    uvd: UVData object
        Input data to compute covariance matrix for.
        Only supports 1d arrays.
    antenna_chromaticity: float, optional
        intrinsic chromaticity of each antenna and the sky (in ns).
        Default is 0.0
    bl_cutoff_buffer: float, optional
        For two baselines with length b2 > b1
        baselines with (b1 + bl_cutoff_buffer) * max_freq < (b2 * min_freq)
        (max u of b1 < min u of b2) are set to have covariances of zero.
        thus the larger the bl_cutoff_buffer_buffer, the fewer off-diag terms are set to zero.
    order_by_bl_length: bool, optional
        If true, order columns and rows by increasing baseline length.
    return bl_len_freqs: bool, optional
        if True, return vector of baseline lengths and frequencies.
    array_config_kwargs: kwarg dict
        kargs for simulator.initialize_simulation_uvdata
        see docstring.


    Returns
    -------
    if bl_len_freqs:
        blvals: array-like
            (Nbls * Nfreqs) vector of baseline lengths for each cell.
        nuvals: array-like
            (Nbls * Nreqs) vector of frequency vals for each cell.
        covmat: array-like
            (Nbls * Nfreqs, Nbls * Nfreqs) filtering matrix that can be used to
            cut between the fabric of the foregrounds and cosmological signal.
    else:
        covmat: array-like
            (Nbls * Nfreqs, Nbls * Nfreqs) filtering matrix that can be used to
            cut between the fabric of the foregrounds and cosmological signal.

    """
    if uvd is None:
        _, _, uvd = simulator.initialize_simulation_uvdata(**array_config_kwargs)
    for i in range(1, 2):
        if np.any(~np.isclose(uvd.uvw_array[:, i], 0.0)):
            raise NotImplementedError("cov_mat_simple only currently supports 1d arrays oriented EW.")
    # only works on a 1d array.
    data_inds = np.where(uvd.time_array == np.unique(uvd.time_array)[0])[0]
    # sort data inds by baseline length if we wish.
    if order_by_bl_length:
        data_inds = data_inds[np.argsort(np.abs(uvd.uvw_array[data_inds, 0]))]
    blvals = np.outer(uvd.uvw_array[data_inds, 0], np.ones_like(uvd.freq_array[0])).flatten()
    nuvals = np.outer(np.ones(uvd.Nbls), uvd.freq_array[0]).flatten()
    u_x, u_y = np.meshgrid(blvals * nuvals / 3e8, blvals * nuvals / 3e8)
    nu_x, nu_y = np.meshgrid(nuvals, nuvals)
    covmat = np.sinc(2 * (u_x - u_y)) * np.sinc(2 * antenna_chromaticity * (nu_x - nu_y))
    if np.isfinite(bl_cutoff_buffer):
        del nu_x, u_x
        del nu_y, u_y
        min_freq = uvd.freq_array.min()
        max_freq = uvd.freq_array.max()
        for i, bi in enumerate(np.abs(blvals)):
            for j, bj in enumerate(np.abs(blvals)):
                if (min(bi, bj) + bl_cutoff_buffer) * max_freq < max(bi, bj) * min_freq:
                        covmat[i, j] = 0.

    if return_bl_lens_freqs:
        return blvals, nuvals, covmat
    else:
        return covmat

def cov_airy_integral():
    """
    """


def cov_element_airy(signal_frequency_covariance=None):
    """Covariance matrix between airy-beams and flat foregrounds.

    signal_frequency_covariance: function nu_1, nu_2 -> covariance, optional.
    """

def cov_mat_gsm_simulated():
    """Estimate a bootstrapped gsm covariance matrix using random rotations.
    """
    return

def cov_mat_eor_simulated():
    """Estimate a bootstrapped eor covariance matrix using random gaussian draws.
    """
    return
