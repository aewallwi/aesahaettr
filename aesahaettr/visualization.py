import numpy as np
import copy
from uvtools import dspec
from . import defaults

def delay_transform_sort_by_baseline_length(uvd, tind=0, polind=0, window='bh', min_bl=0.1):
    """
    generate an (NF x NBL) array with baselines arranged in ascending length

    Parameters
    ----------
    uvd: UVData object
        UVData with data that you want to inspect FFT of.
    polind: int, optional
        index of polarization to return.
        default is 0.
    window: str, optional
        fourier transform window function.
        default is blackman harris.
    min_bl: float, optional
        minimum baseline length to include
        default is 0.1 (meters)

    Returns
    -------
    bl_lens: array-like
        Nbls float array with lengths of each baseline.
    delays: array-like
        Nfreqs float array of delays for each baseline.
    fftd_data: array-like
        Nbls x Nfreqs array of data that has been Fourier transformed along the
        frequency axis. Baselines are arranged along baseline axis in ascending length.
    """
    data_inds = np.where(uvd.time_array == np.unique(uvd.time_array)[tind])[0]
    data = uvd.data_array[data_inds, 0, :, polind].reshape(len(data_inds), uvd.Nfreqs)
    bl_lens = np.linalg.norm(uvd.uvw_array[data_inds], axis=1)
    sorted_lens = np.argsort(bl_lens)
    data = data[sorted_lens, :]
    bl_lens = bl_lens[sorted_lens]
    data = data[bl_lens > min_bl]
    bl_lens = bl_lens[bl_lens > min_bl]
    wf = dspec.gen_window(window, uvd.Nfreqs)
    delays = 1e9 * np.fft.fftshift(np.fft.fftfreq(uvd.Nfreqs, np.mean(np.diff(uvd.freq_array))))
    fftd_data =  np.fft.fftshift(np.fft.fft(np.fft.fftshift(wf[None, :] * data, axes=1), axis=1), axes=1)
    return bl_lens, delays, fftd_data
