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
import numba_scipy
import scipy.special as sp
import scipy.sparse as sparse
from multiprocessing import Pool
import tensorflow as tf
from uvtools import dspec

def convert_to_sparse_bands(cov_matrix):
    """convert covariance matrix to a sparse banded matrix (if possible)

    Parameters
    cov_matrix: array-like
        Ndata x Ndata square matrix

    Returns
    -------
    dia_matrix: scipy.sparse.dia_matrix
        scipy.sparse.dia_matrix derived from non-zero bands
    """
    diagonals = []
    offsets = []
    for k in range(cov_matrix.shape[0]):
        band = [cov_matrix[i, i + k] for i in range(cov_matrix.shape[0] - k)]
        if np.any(np.abs(np.asarray(band)) > 0.):
            diagonals.append(band)
            offsets.append(k)
    for k in range(1, cov_matrix.shape[1]):
        band = [cov_matrix[i + k, i] for i in range(cov_matrix.shape[1] - k)]
        if np.any(np.abs(np.asarray(band)) > 0.):
            diagonals.append(band)
            offsets.append(-k)
    dia_matrix = sparse.diags(diagonals, offsets)
    return dia_matrix


def cov_mat_simple(uvd=None, antenna_chromaticity=0.0, bl_cutoff_buffer=np.inf, order_by_bl_length=False,
                   return_bl_lens_freqs=False, return_uvdata=False, intra_baseline_only=False, **array_config_kwargs):
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
    intra_baseline_only: bool, optional
        if True, only keep blocks corresponding to the same baseline.
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
        if return_uvdata:
            uvd: UVData object containing metadata information.
    else:
        covmat: array-like
            (Nbls * Nfreqs, Nbls * Nfreqs) filtering matrix that can be used to
            cut between the fabric of the foregrounds and cosmological signal.
        uvd: UVData object containing metadata information.
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
    if intra_baseline_only:
        covmat[~np.isclose(u_x / nu_x * 3e8, u_y / nu_y * 3e8)] = 0.
    if return_bl_lens_freqs:
        if return_uvdata:
            return blvals, nuvals, covmat, uvd
        else:
            return blvals, nuvals, covmat
    else:
        if return_uvdata:
            return covmat, uvd
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


def cov_matrix_airy(uvdata=None, output_dir='./', mode='foregrounds', correlated_freqs=True,
                    clobber=True, order_by_bl_length=False, bl_cutoff_buffer=np.inf, return_uvdata=False,
                    parallelize=False,
                    **array_config_kwargs):
    """Covariance for flat-spectrum unclustered sources viewed by an array with an airy beam.

    Parameters
    ----------
    output_dir: str, optional
        where to write template container
    correlated_freqs: bool, optional
        if true, assume that frequencies are correlated.
    nside_sky: int, optional
        nsides of healpix sky to simulate.
    clobber: bool, optional
        if true, overwrite existing files. If not, dont.
        only applies to templaet data.
    return uvdata: bool, optional
        if true, return uvdata object as well

    Returns
    -------
    cov-mat: array-like
        covariance matrix that is (Nfreqs * Nbls) x (Nfreqs * Nbls)
        derived from randomly drawing a simulated sky.
    if return_uvdata:
        uvdata: UVData
            UVData object with all of the metadata / data shape.
    """
    if 'antenna_diameter' not in array_config_kwargs:
        array_config_kwargs['antenna_diameter'] = defaults.antenna_diameter
    if uvdata is None:
        uvdata, _, _ = visibilities.initialize_uvdata(**array_config_kwargs)

    data_inds = np.where(uvdata.time_array == uvdata.time_array[0])[0]
    nblfrqs = uvdata.Nbls * uvdata.Nfreqs
    if order_by_bl_length:
        data_inds = data_inds[np.argsort(np.abs(uvdata.uvw_array[data_inds, 0]))]
    blvals = np.outer(uvdata.uvw_array[data_inds, 0], np.ones_like(uvdata.freq_array[0])).flatten()
    nuvals = np.outer(np.ones(uvdata.Nbls), uvdata.freq_array[0]).flatten()
    nx = len(blvals)
    min_freq = uvdata.freq_array.min()
    max_freq = uvdata.freq_array.max()
    if not parallelize:
        covmat = np.zeros((nx, nx))
        for i, j in itertools.combinations(range(nx), 2):
            if correlated_freqs or nuvals[i] == nuvals[j]:
                if (min(blvals[i], blvals[j]) + bl_cutoff_buffer) * max_freq < max(blvals[i], blvals[j]) * min_freq:
                    covmat[i, j] = 0.
                    covmat[j, i] = 0.
                else:
                    covmat[i, j] = 2 * np.pi * integrate.quad(airy_cov_integrand, 0, np.pi / 2.,
                                                              args=(nuvals[i], nuvals[j], blvals[i], blvals[j], array_config_kwargs['antenna_diameter']))[0]
                    covmat[j, i] = covmat[i, j]
        for i in range(nx):
            covmat[i, i] = 2 * np.pi * integrate.quad(airy_cov_integrand, 0, np.pi / 2.,
                                                      args=(nuvals[i], nuvals[i], blvals[i], blvals[i], array_config_kwargs['antenna_diameter']))[0]
    else:
        raise NotImplementedError("parallelized integral calculation not yet implemented but needs to be!")
        pool = Pool()
        tasks = []
    if return_uvdata:
        return covmat, uvdata
    else:
        return covmat

def process_matrix_element(bl1, bl2, nu1, nu2, min_freq, max_freq, antenna_diameter, correlated_freqs=True):
    if correlated_freqs or nu1 == nu2:
        if (min(bl1, bl2) + bl_cutoff_buffer) * max_freq < max(bl1, bl2) * min_freq:
            return 0.0
        else:
            return 2 * np.pi * integrate.quad(airy_cov_integrand, 0, np.pi / 2.,
                                              args=(nu1, nu2, bl1, bl2, antenna_diameter))[0]
    else:
        return 0.0

def cov_mat_simulated(ndraws=1000, compress_by_redundancy=False, output_dir='./', mode='gsm',
                     nside_sky=defaults.nside_sky, clobber=True, order_by_bl_length=False,
                     return_uvdata=False, **array_config_kwargs):
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
    return uvdata: bool, optional
        if true, return uvdata object as well

    Returns
    -------
    cov-mat: array-like
        covariance matrix that is (Nfreqs * Nbls) x (Nfreqs * Nbls)
        derived from randomly drawing a simulated sky.
    if return_uvdata:
        UVData object with all of the metadata / data shape.
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
    if return_uvdata:
        if compress_by_redundancy:
            return uvdata_compressed, cov_mat
        else:
            return cov_mat, uvdata
    else:
        return cov_mat


def dpss_modeling_vectors(uvdata, eigenval_cutoff=1e-10, antenna_diameter=defaults.antenna_diameter):
    """Generate per-baseline dpss eigenvectors for a uvdata object.

    Parameters
    ----------
    uvdata: UVData object
        template UVData object to base eigenvectors on.
    eigenval_cutoff: float, optional
        order of eigenvalue to cutoff DPSS modeling vectors at.
    antenna_diameter: float, optional
        aperture size. used to set intrinsic antenna chromaticity.
    Returns
    -------
    dpss_vectors: array-like
        (Nbls x Nfreqs) x Nvecs array of DPSS vectors with freq raveled inside of baseline
        these vectors are zero outside of their baseline block.
    """
    # generate dpss vectors
    dpss_vectors = []
    for ant1, ant2, bldly in zip(uvdata.ant_1_array, uvdata.ant_2_array, (np.linalg.norm(uvdata.uvw_array, axis=1) + antenna_diameter)/ 3e8):
        blinds = uvdata.antpair2ind(ant1, ant2)
        bl_dpss_vectors = dspec.dpss_operator(x=uvdata.freq_array[0], filter_centers=[0.], filter_half_widths=[bldly], eigenval_cutoff=eigenval_cutoff)[1]
        # pad zeros representing data outside of the vectors particular baseline block.
        bl_dpss_vectors = np.pad(bl_dpss_vectors, [(blinds[0], uvdata.Nblts - blinds[-1]), (0, 0)])
        dpss_vectors.append(bl_dpss_vectors)
    dpss_vectors = np.vstack(dpss_vectors)
    return dpss_vectors


def cov_mat_simple_evecs(uvdata=None, eigenval_cutoff=1e-10, use_sparseness=False, eig_kwarg_dict=None, use_tensorflow=False,
                         write_outputs=False, output_dir='./', **cov_mat_simple_kwargs):
    """Generate eigenvectors of cov_mat_simple covariance to fit data.

    Parameters
    ----------
    eigenval_cutoff: float, optional
        include eigenvectors with eigenvals above this fraction of max eigenval.
    use_sparseness: bool, optional
        treate covariance as sparse banded matrix to speed up eigenvector estimation
        requires cov_mat_simple_kwargs['bl_cutoff_buffer'] finite.
    eig_kwarg_dict: dictionary, optional
        dict of kwargs for scipy.sparse.linalg.eigsh or np.linalg.eigh.
    use_tensorflow: bool, optional
        use tensorflow and automatic GPU acceleration.
        default is False.
    write_outputs: bool, optional
        if True, write out eigenvectors and eigenvalues
        default is False.
    output_dir: str, optional
        location to write outputs if write_outputs=True
        default is './'
    cov_mat_simple_kwargs: kwargs
        kwargs for cov_mat_simple except for return_bl_lens_freqs and order_by_bl_length
        See cov_mat_simple docstring.
    Returns
    -------
    evals: array-like
        Array of eigenvalues.
    evecs: array-like
        (Nbls x Nfreqs) x Nvecs array where each slice in the 0th dim
        is an eigenvector of cov_mat_simple reshaped into the proper uvdata shape.
    """
    if eig_kwarg_dict is None:
        eig_kwarg_dict = {}
    if uvdata is None:
        cmat, uvdata = cov_mat_simple(return_uvdata=True, return_bl_lens_freqs=False, order_by_bl_length=False, **cov_mat_simple_kwargs)
    else:
        cmat = cov_mat_simple(uvd=uvdata, return_uvdata=False, return_bl_lens_freqs=False, order_by_bl_length=False, **cov_mat_simple_kwargs)
    if use_sparseness and 'bl_cutoff_buffer' in cov_mat_simple_kwargs and np.isfinite(cov_mat_simple_kwargs['bl_cutoff_buffer']):
        cmat = convert_to_sparse_bands(cmat)
        evals, evecs = sparse.linalg.eigsh(cmat, k=cmat.shape[0] // 2, **eig_kwarg_dict)
    elif use_tensorflow:
        cmat = tf.convert_to_tensor(cmat.astype(np.float64))
        evals, evecs = tf.linalg.eigh(cmat)
        evals = evals.numpy()
        evecs = evecs.numpy()
    else:
        evals, evecs = np.linalg.eigh(cmat, **eig_kwarg_dict)
    evalorder = np.argsort(evals)[::-1]
    evals = evals[evalorder]
    evecs = evecs[:, evalorder]
    to_keep = evals >= evals.max() * eigenval_cutoff
    evecs = evecs[:, to_keep]
    evals = evals[to_keep]
    # reshape evecs to uvdata data_array
    if write_outputs:
        basename = visibilities.get_basename(**cov_mat_simple_kwargs)
        np.savez(os.path.join(output_dir, basename + f'_simple_cov_evecs_evalcut_{10*np.log10(eigenval_cutoff):.1f}dB.npz'), evecs=evecs)

    return evals, evecs
