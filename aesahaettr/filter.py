import copy
from uvtools import dspec
import numpy as np
from .  import covariances

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
    cmat_simple = covariances.cov_mat_simple(uvd, **cov_kwargs)
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
                  * uvd.Nfreqs))).reshape(len(data_inds), uvd.Nfreqs)
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
