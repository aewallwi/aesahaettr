# tools for computing ivsibility covariances.

import numpy as np
from . import visibilities
from . import defaults
import tqdm
import healpy as hp
# import airy beam model.
from hera_sim.visibilities import vis_cpu
import numba
import scipy.integrate as integrate
import itertools
import scipy.special as sp
import numba_special

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
        intrinsic chromaticity of each antenna and the sky (in seconds).
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
        kargs for visibilities.initialize_uvdata
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
        uvd, _, _ = visibilities.initialize_uvdata(**array_config_kwargs)
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


@numba.jit(nopython=True)
def airy_cov_integrand(theta, nu1, nu2, baseline1, baseline2, antenna_diameter=defaults.antenna_diameter):
    """Integrand in covariance between two baselines with airy-beams and spatially uncorrelated flat-spectrum sources.

    Parameters
    ----------
    theta: float
        direction on the sky.
    nu1: float
        frequency of first baseline (MHz)
    nu2: float
        frequency of second baseline
    baseline1: float
        EW length of baseline1
    baseline2: float
        EW length of baseline2
    antenna_diameter: float, optional
        diameter of antennas dermining airy beam.

    Returns
    -------
    integrand:  float
        A(\hat{s}, nu_1) A(hat{s}, nu_2) e^{-2 pi I (u_1 -u_2) \cdot \hat{s}} f

    """
    x1 = np.sin(theta) * 2 / antenna_diameter * 3e8 / nu1
    x2 = np.sin(theta) * 2 / antenna_diameter * 3e8 / nu2
    if x1 > 0:
        airy1 = (2 * sp.j1(x1) / x1) ** 2.
    else:
        airy1 = 1.
    if x2 > 0:
        airy2 = (2 * sp.j1(x2) / x2) ** 2.
    else:
        airy2 = 1.
    du = np.abs(baseline1 * nu1 / 3e8 - baseline2 * nu2 / 3e8)
    integrand = airy1 * airy2 * sp.j0(du * np.sin(theta)) * np.sin(theta)
    return integrand


def loop_over_cov_matrix(blvals, nuvals, correlated_freqs=True, antenna_diameter=defaults.antenna_diameter):
    nx = len(blvals)
    covmat = np.zeros((nx, nx))
    for i, j in itertools.combinations(range(nx), 2):
        if correlated_freqs or nuvals[i] == nuvals[j]:
            #covmat[i, j] = cov_airy_integral(nuvals[i], nuvals[j], blvals[i], blvals[j],
            #                                 antenna_diameter=antenna_diameter)
            covmat[i, j] = 2 * np.pi * integrate.quad(airy_cov_integrand, 0, np.pi / 2.,
                                                      args=(nuvals[i], nuvals[j], blvals[i], blvals[j], antenna_diameter))[0]
            covmat[j, i] = covmat[i, j]
    for i in range(nx):
        covmat[i, i] = 2 * np.pi * integrate.quad(airy_cov_integrand, 0, np.pi / 2.,
                                                  args=(nuvals[i], nuvals[i], blvals[i], blvals[i], antenna_diameter))[0]
    return covmat


def cov_matrix_airy(compress_by_redundancy=False, output_dir='./', mode='foregrounds', correlated_freqs=True,
                    clobber=True, order_by_bl_length=False, **array_config_kwargs):
    """Covariance for flat-spectrum unclustered sources viewed by an array with an airy beam.

    Parameters
    ----------
    compress_by_redundancy: bool, optional
        if True, only compute covariance for one baseline per redundant group.
    output_dir: str, optional
        where to write template container
    correlated_freqs: bool, optional
        if true, assume that frequencies are correlated.
    nside_sky: int, optional
        nsides of healpix sky to simulate.
    clobber: bool, optional
        if true, overwrite existing files. If not, dont.
        only applies to templaet data.

    Returns
    -------
    cov-mat: array-like
        covariance matrix that is (Nfreqs * Nbls) x (Nfreqs * Nbls)
        derived from randomly drawing a simulated sky.

    """
    if 'antenna_diameter' not in array_config_kwargs:
        array_config_kwargs['antenna_diameter'] = defaults.antenna_diameter
    uvdata, beams, beam_ids = visibilities.initialize_uvdata(output_dir=output_dir, clobber=clobber,
                                                             **array_config_kwargs)
    if compress_by_redundancy:
        uvdata_compressed = uvdata.compress_by_redundancy(tol = 0.25 * 3e8 / uvdata.freq_array.max(), inplace=False)
        nblfrqs = uvdata_compressed.Nbls * uvdata_compressed.Nfreqs
        data_inds = np.where(uvdata_compressed.time_array == uvdata.time_array[0])[0]
        if order_by_bl_length:
            data_inds = data_inds[np.argsort(np.abs(uvdata_compressed.uvw_array[data_inds, 0]))]
        blvals = np.outer(uvdata_compressed.uvw_array[data_inds, 0], np.ones_like(uvdata_compressed.freq_array[0])).flatten()
        nuvals = np.outer(np.ones(uvdata_compressed.Nbls), uvdata_compressed.freq_array[0]).flatten()
    else:
        data_inds = np.where(uvdata.time_array == uvdata.time_array[0])[0]
        nblfrqs = uvdata.Nbls * uvdata.Nfreqs
        if order_by_bl_length:
            data_inds = data_inds[np.argsort(np.abs(uvdata.uvw_array[data_inds, 0]))]
        blvals = np.outer(uvdata.uvw_array[data_inds, 0], np.ones_like(uvdata.freq_array[0])).flatten()
        nuvals = np.outer(np.ones(uvdata.Nbls), uvdata.freq_array[0]).flatten()
    cov_mat = loop_over_cov_matrix(blvals, nuvals, correlated_freqs=correlated_freqs,
                                   antenna_diameter=array_config_kwargs['antenna_diameter'])
    return cov_mat


def cov_mat_simulated(ndraws=1000, compress_by_redundancy=False, output_dir='./', mode='gsm',
                     nside_sky=defaults.nside_sky, clobber=True, order_by_bl_length=False,
                     **array_config_kwargs):
    """Estimate a bootstrapped gsm covariance matrix using random rotations of GlobalSkyModel.

    Parameters
    ----------
    ndraws: int, optional
        number of realizations of the sky to derive covariance from.
        default is 1000
    compress_by_redundancy: bool, optional
        if True, only compute covariance for one baseline per redundant group.
    output_dir: str, optional
        where to write template container
    mode: str, optional
        'gsm' for gsm or 'eor' for random fluctuations.
    nside_sky: int, optional
        nsides of healpix sky to simulate.
    clobber: bool, optional
        if true, overwrite existing files. If not, dont.
        only applies to templaet data.

    Returns
    -------
    cov-mat: array-like
        covariance matrix that is (Nfreqs * Nbls) x (Nfreqs * Nbls)
        derived from randomly drawing a simulated sky.
    """
    uvdata, beams, beam_ids = visibilities.initialize_uvdata(output_dir=output_dir, clobber=clobber,
                                                **array_config_kwargs)
    if mode == 'gsm':
        signalcube = visibilities.initialize_gsm(uvdata.freq_array[0], nside_sky=nside_sky)
    if compress_by_redundancy:
        uvdata_compressed = uvdata.compress_by_redundancy(tol = 0.25 * 3e8 / uvdata.freq_array.max(), inplace=False)
        nblfrqs = uvdata_compressed.Nbls * uvdata_compressed.Nfreqs
        data_inds = np.where(uvdata_compressed.time_array == uvdata.time_array[0])[0]
        if order_by_bl_length:
            data_inds = data_inds[np.argsort(np.abs(uvdata_compressed.uvw_array[data_inds, 0]))]
    else:
        data_inds = np.where(uvdata.time_array == uvdata.time_array[0])[0]
        nblfrqs = uvdata.Nbls * uvdata.Nfreqs
        if order_by_bl_length:
            data_inds = data_inds[np.argsort(np.abs(uvdata.uvw_array[data_inds, 0]))]

    cov_mat = np.zeros((nblfrqs, nblfrqs), dtype=complex)
    mean_mat = np.zeros(nblfrqs, dtype=complex)

    for draw in tqdm.tqdm(range(ndraws)):
        # generate random rotation
        if mode == 'gsm':
            rot = hp.Rotator(rot=(np.random.rand() * 360, (np.random.rand() * 180 - 90),
                                  np.random.rand() * 360))
            signalcube = np.asarray(rot.rotate_map_pixel(signalcube))
        else:
            signalcube = visibilities.initialize_eor(uvdata.freq_array[0], nside_sky=nside_sky)
        uvdata.data_array[:] = 0.0
        simulator = vis_cpu.VisCPU(uvdata=uvdata, sky_freqs=uvdata.freq_array[0], beams=beams,
                                       beam_ids=beam_ids, sky_intensity=signalcube)
        simulator.simulate()
        if compress_by_redundancy:
            uvdata_compressed = simulator.uvdata.compress_by_redundancy(tol = 0.25 * 3e8 / uvdata.freq_array.max(), inplace=False)
            data_vec = uvdata_compressed.data_array[data_inds, 0, :, 0].flatten()
        else:
            data_vec = simulator.uvdata.data_array[data_inds, 0, :, 0].flatten()
        mean_mat += data_vec
        cov_mat += np.outer(data_vec, np.conj(data_vec))
    mean_mat = mean_mat / ndraws
    cov_mat = cov_mat / ndraws - np.outer(mean_mat, np.conj(mean_mat))
    return cov_mat
