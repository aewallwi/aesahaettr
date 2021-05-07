import pytest
from .. import simulator
import os
import yaml
import numpy as np


def test_get_basename():
    basename = simulator.get_basename()
    assert basename == 'HERA-III_antenna_diameter2.0_fractional_spacing1.0_nant10_nf200_df100.000kHz_f0100.000MHz'

def test_initialize_telescope_yaml(tmpdir):
    tmppath = tmpdir.strpath
    obs_param_yaml_name, telescope_yaml_name, csv_name = simulator.initialize_telescope_yamls(output_dir=tmppath, df=50e3)
    assert os.path.exists(obs_param_yaml_name)
    assert os.path.exists(telescope_yaml_name)
    assert os.path.exists(csv_name)

def test_intialize_simulation_uvdata(tmpdir):
    tmppath = tmpdir.strpath
    array_kwargs = {'df': 50e3, 'f0': 831.7e6, 'nf': 13, 'antenna_diameter': 4.3, 'fractional_spacing': 3.1, 'antenna_count': 21}
    obs_param_yaml_name, telescope_yaml_name, csv_name = simulator.initialize_telescope_yamls(output_dir=tmppath, **array_kwargs)
    beams, beam_ids, uvdata = simulator.initialize_simulation_uvdata(output_dir=tmppath, keep_config_files_on_disk=False, **array_kwargs)
    # assert that yamls have been erased.
    assert not os.path.exists(obs_param_yaml_name)
    assert not os.path.exists(telescope_yaml_name)
    assert not os.path.exists(csv_name)
    # now check that they have not been erased.
    beams, beam_ids, uvdata = simulator.initialize_simulation_uvdata(output_dir=tmppath, keep_config_files_on_disk=True, **array_kwargs)

    assert os.path.exists(obs_param_yaml_name)
    assert os.path.exists(telescope_yaml_name)
    assert os.path.exists(csv_name)

    assert np.allclose(np.diff(uvdata.freq_array[0]), 50e3)
    # check that uvdata parameters make sense.
    assert uvdata.Nfreqs == 13
    assert uvdata.freq_array.min() == 831.7e6
    assert uvdata.Nbls == 21 * 22 / 2
    assert uvdata.Ntimes == 1
    bl_lens = np.linalg.norm(uvdata.uvw_array, axis=1)
    gtz = bl_lens > 0.0
    assert np.isclose(np.min(bl_lens[gtz]), 4.3 * 3.1)
    assert np.isclose(np.max(bl_lens), 4.3 * 3.1 * 333)
