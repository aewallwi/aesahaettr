import numpy as np
from .. import filter_simulator
import pytest



def test_simulate_and_filter(tmpdir):
    tmppath = tmpdir.strpath
    array_kwargs={'nf': 13, 'f0': 831.7e6, 'antenna_count': 5, 'compress_by_redundancy':True,
                  'antenna_diameter': 2.1, 'fractional_spacing': 4.3}
    uvd_total, uvd_filtered = filter_simulator.simulate_and_filter(output_dir=tmppath, clobber=True, **array_kwargs)
    assert uvd_total.Nfreqs == 13
    assert uvd_total.freq_array.min() == 831.7e6
    assert uvd_total.Nbls == 5 * 6 / 2 - 4
    assert uvd_total.Ntimes == 1
    bl_lens = np.linalg.norm(uvd_total.uvw_array, axis=1)
    gtz = bl_lens > 0.0
    assert np.isclose(np.min(bl_lens[gtz]), 2.1 * 4.3)
    assert np.isclose(np.max(bl_lens), 2.1 * 4.3 * 11)
    # check that uvd_total has significantly larger RMS data.
    assert np.sqrt(np.mean(np.abs(uvd_total.data_array)**2.))\
           >= 1e-2 * np.sqrt(np.mean(np.abs(uvd_filtered.data_array) ** 2.))
