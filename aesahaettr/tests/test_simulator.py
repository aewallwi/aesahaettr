import pytest
from .. import simulator
import os
import yaml


def test_get_basename():
    basename = simulator.get_basename()
    assert basename == 'HERA-III-antenna_diameter2.0_fractional_spacing1.0_G10_nf200_df100.000kHz_f0100MHz'

def test_initialize_telescope_yaml(tmpdir):
    str_path = tmpdir.pathstr
    obs_param_yaml_name, telescope_yaml_name, csv_name = simulator.test_initialize_telescope_yamls(output_dir=tmpdir, df=50e3)
    assert os.path.exists(obs_param_yaml_name)
    assert os.path.exists(telescope_yaml_name)
    assert os.path.exists(csv_name)

def test_intialize_simulation_uvdata(tmpdir):
    
