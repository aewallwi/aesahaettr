# tools for computing ivsibility covariances.

import numpy as np
from . import visibilities
from . import defaults
import tqdm
import healpy as hp
# import airy beam model.
from hera_sim.visibilities import vis_cpu
import numba
import scipy.integrate

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


def analytic_airy(theta, nu, antenna_diameter=defaults.antenna_diameter):
    """An airy beam.

    Parameters
    ----------
    theta: float
        zenith angle (radians)
    nu: float
        frequency (Hz)
    antenna_diameter: float
        diameter of antenna (meters)
        see defaults. antenna_diameter

    Returns
    -------
    gain: float
        peak normalized directivity of an analytic airy beam
        at frequency nu and angle theta from boresight.
    """
    x = np.sin(theta) * 2 / antenna_diameter * 3e8 / nu
    if np.abs(x) > 0:
        return (2 * sp.jn(1, x) / x) ** 2.
    else:
        return 1.


def cov_airy_integral(nu1, nu2, baseline1, baseline2, antenna_diameter=defaults.antenna_diameter):
    """Compute beam-baseline integral for pair of beams.

    Parameters
    ----------
    nu1, first frequency
    nu2, second frequency
    baseline1, length of first baseline
    baseline2, length of second baseline
    antenna_diameter, diameter of antennas

    Returns
    -------
    integral, float
        \int A(x, nu_1) A^*(x, nu_2) sin(x) e^(-2 \pi I (u_1 - u_2)) d\Omega
        where A(x, nu_1) is given by an airy beam with diameter antenna_diameter
    """
    integrand = lambda x: analytic_airy(x, nu1, antenna_diameter) * analytic_airy(x, nu2, antenna_diameter) * \
                                           sp.jn(0, np.linalg.norm(baseline1 * nu1 / 3e8 - baseline2 * nu2 / 3e8) * np.sin(x)) * np.sin(x)
    integral = 2 * np.pi * scipy.integrate(integrand, 0, np.pi / 2.)



def cov_element_airy(signal_frequency_covariance=None):
    """Covariance matrix between airy-beams and flat foregrounds.

    signal_frequency_covariance: function nu_1, nu_2 -> covariance, optional.
    """

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

def cov_mat_eor_simulated(nsamples=1000):
    """Estimate a bootstrapped eor covariance matrix using random gaussian draws.
    """

    covmat += np.outer(vis_sample, np.conj(vis_sample))
    meanmat += vis_sample

    return
