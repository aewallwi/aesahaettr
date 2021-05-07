# tools for computing ivsibility covariances.

import numpy as np


def cov_mat_simple(uvd=None, antenna_chromaticity=0.0, bl_cutoff_buffer=np.inf, order_by_bl_length=False,
                   sim_param_yaml=None, return_bl_lens_freqs=False):
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
    sim_param_yaml: str, optional
        used if uvd is None.
        parameters to generate a uvdata object to compute simple covariance matrix for.
    return bl_len_freqs: bool, optional
        if True, return vector of baseline lengths and frequencies..

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
        if sim_param_yaml is not None:
            uvd, _, _ = initialize_uvdata_from_params(obs_param_yaml_name)
            uvd = _complete_uvdata(uvd)
        else:
            raise ValueError("Must provide a uvdata object or sim_param_yaml")
    for i in range(1, 2):
        if np.any(~np.isclose(uvd.uvw_array[:, i], 0.0)):
            raise NotImplementedError("cov_mat_simple only currently supports 1d arrays oriented EW.")
    # only works on a 1d array.
    data_inds = np.where(uvd.time_array == np.unique(uvd.time_array)[0])[0]
    # sort data inds by baseline length if we wish.
    if order_by_bl_length:
        data_inds[np.argsort(np.abs(uvd.uvw_array[data_inds, 0]))]
    blvals = np.outer(uvd.uvw_array[data_inds, 0], np.ones_like(uvd.freq_array[0])).flatten('F')
    nuvals = np.outer(np.ones(uvd.Nbls), uvd.freq_array[0]).flatten('F')
    u_x, u_y = np.meshgrid(blvals * nuvals / 3e8, blvals * nuvals / 3e8)
    nu_x, nu_y = np.meshgrid(nuvals, nuvals)
    covmat = np.sinc(2 * (u_x - u_y)) * np.sinc(2 * antenna_chromaticity * (nu_x - nu_y))
    if np.isfinite(bl_cutoff_buffer):
        del nu_x, u_x
        del nu_y, u_y
        bl_x, bl_y = np.meshgrid(blvals, blvals)
        min_freq = uvd.freq_array.min()
        max_freq = uvd.freq_array.max()
        for i, bi in enumerate(b_x):
            for j, bj in enumerate(b_y):
                if (min(bi, bj) + bl_cutoff_buffer) * max_freq < max(bi, bj) * min_freq:
                        covmat[i, j] = 0.

    if return_bl_lens_freqs:
        return blvals, nuvals, covmat
    else:
        return covmat