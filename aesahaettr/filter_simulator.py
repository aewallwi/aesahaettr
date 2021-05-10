
import numpy as np
from . import visibilities
from . import filter
import os
import argparse
import copy

def simulate_and_filter(tol=1e-11, output_dir='./', clobber=False,
                        compress_by_redundancy=False,
                        bl_cutoff_buffer=np.inf, intrinsic_chromaticity_multiplier=1.0,
                        use_dayenu=False, **sim_kwargs):
    """Simulate array configuration and apply filtering.

    Parameters
    ----------
    tol: float, optional
        amount to suppress foregrounds by in matrix inversion.
        default is 1e-11.
    output_dir: str, optional
        directory to store outputs in.
        default is './'
    clobber: bool, optional
        overwrite files if they exist.
    compress_by_redundancy: bool, optional
        if true, only compute and filter
        single baseline per redundant group.
        default is False
    bl_cutoff_buffer: float, optional
        see docstring for covariances.cov_mat_simple.
        default is np.inf.
    intrinsic_chromaticity_multiplier: float, optional
        will filter frequency modes up to intrinsice chromaticity of dish set by
        antenna_diameter / C  * intrinsic_chromaticity_multiplier
    use_dayenu: bool, optional
        if true, use the per-baseline dayenu filter on data instead of
        full inter-baseline filter.
        invalidates bl_cutoff_buffer.
    sim_kwargs: additional optional params for simulation.
        parameters of the sky simulation. See compute_visibilities docstring.
    """
    if 'antenna_diameter' in sim_kwargs:
        antenna_diameter = sim_kwargs['antenna_diameter']
    else:
        antenna_diameter = 2.0
    basename = visibilities.get_basename(**sim_kwargs)
    filtered_pbl_file = os.path.join(output_dir, basename + f'_bmult{intrinsic_chromaticity_multiplier:.2f}_tol{tol:.1e}_pbl.uvh5')
    filtered_ibl_file = os.path.join(output_dir, basename + f'_bmult{intrinsic_chromaticity_multiplier:.2f}_tol{tol:.1e}_ibl.uvh5')
    if (not os.path.exists(filtered_pbl_file) or not os.path.exists(filtered_ibl_file)) or not skip_existing:
        uvd_eor_name = os.path.join(output_dir, basename + '_eor.uvh5')
        uvd_gsm_name = os.path.join(output_dir, basename + '_gsm.uvh5')
        if os.path.exists(uvd_eor_name) and os.path.exists(uvd_gsm_name) and not clobber:
            uvd_eor = UVData()
            uvd_eor.read_uvh5(uvd_eor_name)
            uvd_gsm = UVData()
            uvd_gsm.read_uvh5(uvd_gsm_name)
        else:
            uvd_eor, uvd_gsm = visibilities.compute_visibilities(output_dir=output_dir, clobber=clobber, **sim_kwargs)
        uvd_total = copy.deepcopy(uvd_eor)
        uvd_total.data_array = uvd_eor.data_array + uvd_gsm.data_array
        uvd_filtered = filter.filter_data(uvd_total, eta_max=antenna_diameter / 3e8 * intrinsic_chromaticity_multiplier,
                                          tol=tol, use_dayenu=use_dayenu, bl_cutoff_buffer=bl_cutoff_buffer)
    return uvd_total, uvd_filtered,


def get_simulation_parser():
    ap = argparse.ArgumentParser(description="Perform a 1-dimensional filtering simulation.")
    simgroup = ap.add_argument_group("Simulation parameters.")
    simgroup.add_argument("--antenna_count", type=int, help="Number of antennas in array simulation.", default=10)
    simgroup.add_argument("--max_bl_length", type=int, help="Maximum baseline length. Can be used to automatically determine the antenna count.", default=None)
    simgroup.add_argument("--antenna_diameter", type=float, default=4.0, help="Diameter of a single antenna element (uses Airy beam).")
    simgroup.add_argument("--fractional_spacing", type=float, help="Distance between elements as a fraction of dish diameter.", default=1.0)
    simgroup.add_argument("--Nfreqs", type=int, help="Number of frequency channels.", default=100)
    simgroup.add_argument("--freq_channel_width", type=float, help="Frequency Channel width [Hz].", default=100e3)
    simgroup.add_argument("--minimum_frequency", type=float, help="Minimum frequency [Hz].", default=100e6)
    filtergroup = ap.add_argument_group("Filtering Parameters")
    filtergroup.add_argument("--tol", type=float, default=1e-11, help="Factor to suppress foregrounds by.")
    filtergroup.add_argument("--buffer_multiplier", type=float, default=1.0, help="Factor to multiply frequency buffer by.")
    filtergroup.add_argument("--per_baseline", default=False, action="store_true", help="Perform per-baseline filter rather then inter-baseline filter.")
    filtergroup.add_argument("--bl_cutoff", default=np.inf, help="Set approximate covariance to be zero between baselines with separation greater then this value.")



def generate_grid_params(r_spaces=None, antenna_diameters=None, nfs=None, antenna_counts=None, dfs=None):

    r_spaces = [1, 1.5, 2.]
    antenna_diameters = [2.0, 4.0, 8.0, 14.0]
    nfs = [100, 200, 400]
    antenna_counts = [10, 15]
    dfs = [100e3, 200e3]
    param_combinations = []

    for fractional_spacing in r_spaces:
        for antenna_diameter in antenna_diameters:
            for nf in nfs:
                for df in dfs:
                    for antenna_count in antenna_counts:
                        param_combinations.append({'fractional_spacing':fractional_spacing, 'antenna_diameter':antenna_diameter,
                                                   'nf':nf, 'df':df, 'antenna_count':antenna_count})
