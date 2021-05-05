import copy
from pyuvdata import
from uvtools import dspec
import numpy as np
filter_defaults = {'antenna_chromaticity': 0.0, bl_cutoff: np.inf, }

def cov_mat_simple(uvd, antenna_chromaticity=0.0, bl_cutoff=np.inf):
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
    bl_cutoff: float, optional
        baselines values separated by greater then bl_cutoff are set to zero.
        this lets us probe the impact of discarding correlations beyond certain baseline separations.
        Default is np.inf (no u_cutoff)
        TODO: get bl_cutoff to be baseline dependent (larger baselines should be more co-variant then shorter baselines).

    Returns
    -------
    covmat: array-like
        (Nbls * Nfreqs, Nbls * Nfreqs) filtering matrix that can be used to
        cut between the fabric of the foregrounds and cosmological signal.

    """
    if np.any(~np.isclose(uvd.uvw_array[:, i])):
        raise NotImplementedError("cov_mat_simple only currently supports 1d arrays oriented EW.")
    # only works on a 1d array.
    data_inds = np.where(uvd.time_array == np.unique(uvd.time_array)[0])[0]
    uvals = np.outer(uvd.uvw_array[data_inds, 0], uvd.freq_array[0] / 3e8).flatten('F')
    nuvals = np.outer(np.ones(uvd.Nbls), uvd.freq_array[0]).flatten('F')
    u_x, u_y = np.meshgrid(uvals, uvals)
    nu_x, nu_y = np.meshgrid(nuvals, nuvals)
    covmat = np.sinc(2 * (u_x - u_y)) * np.sinc(2 * antenna_chromaticity * (nu_x - nu_y))
    if np.isfinite(bl_cutoff):
        cov_mat[np.abs(u_x / nu_x * 3e8 - u_y / nu_y * 3e8) >= bl_cutoff] = 0.
    return covmat

def filter_mat_simple(uvd, tol=1e-9, **cov_kwargs):
    """
    Generate a simple filtering matrix using cov_mat_simple.

    Parameters
    ----------
    uvd: UVData object
        UVData to base filtering matrix off of.
        Currently no support for flagging.
    tol: float, optional
        amount to suppress foreground modes in the simple filter matrix.
        default is 0.0
    cov_kwargs: dict, optional
        keyword arguments for cov_mat_simple. See cov_mat_simple docstring
        for details.

    Returns
    -------
    filter_matrix: array-like
        filtering matrix.


    """
    cmat_simple = cov_mat_simple(uvd, **cov_kwargs)
    cmat_simple = cmat_simple / tol + np.identity(cmat_simple.shape[0])
    return np.linalg.pinv(cmat_simple)

def filter_data(uvd, use_dayenu=False, **filter_kwargs):
    """
    Apply simple filtering to a uvdata object.

    Parameters
    ----------
    use_dayenu: bool, optional
        if True, use dayenu, a per-baseline filter.
        otherwise, use the inter-baseline filter in
        filter_mat_simple.
    filter_kwargs: dict
        filtering arguments. See cov_mat_simple
        docstring and cov_mat_simple docstring for details.

    Returns
    -------
    uvd: uvdata object.
        UVData containing filtered data.

    """
    uvd = copy.deepcopy(uvd)
    if not use_dayenu:
        filter_matrix = filter_mat_simple(uvd, **filter_kwargs)
        for time in np.unique(uvd.time_array):
            data_inds = np.where(uvd.time_array == time)[0]
            for pind in range(uvd.data_array.shape[-1]):
                data = uvd.data_array[data_inds, 0, :, pind].squeeze()
                uvd.data_array[data_inds, 0, :, pind] = \
                (filter_matrix @\
                 (data.reshape(len(data_inds)\
                  * uvd.Nfreqs, order='F'))).reshape(len(data_inds), uvd.Nfreqs, order='F')
    else:
        cache = {}
        for time in np.unique(uvd.time_array):
            data_inds = np.where(uvd.time_array == time)[0]
            for pind in range(uvd.data_array.shape[-1]):
                for rownum in range(uvd.data_array[data_inds, 0, :, pind].squeeze().shape[0]):
                    drow = uvd.data_array[data_inds, 0, :, pind][rownum].squeeze()
                    fw = filter_kwargs['antenna_chromaticity'] + np.linalg.norm(uvd.uvw_array[data_inds[rownum]]) / 3e8
                    filtered\
                    = dspec.dayenu_filter(x=uvd.freq_array[0], data=drow,
                                          wgts=np.ones(uvd.Nfreqs),
                                          cache=cache, filter_centers=[0.0],
                                          filter_half_widths=[fw],
                                          filter_dimensions=[0],
                                          filter_factors=[tol])[0]
                    uvd.data_array[data_inds[rownum], 0, :, pind] = filtered

    return uvd
