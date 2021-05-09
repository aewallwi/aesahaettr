import numpy as np
import itertools
import scipy.special as sp
import copy
import healpy as hp
from pyuvsim import analyticbeam as ab
import os
from . import filter
import yaml
from pygdsm import GlobalSkyModel
from pyuvdata import UVData
from hera_sim.visibilities import vis_cpu
from pyuvsim.simsetup import initialize_uvdata_from_params, _complete_uvdata

golomb_dict = {1:[0], 2:[0,1], 3:[0,1,3],
               4:[0,1,4,6], 5:[0,1,4,9,11],
               6:[0,1,4,10,12,17], 7:[0,1,4,10,18,23,25],
               8:[0,1,4,9,15,22,32,34], 9:[0,1,5,12,25,27,35,41,44],
               10:[0,1,6,10,23,26,34,41,53,55],
               11:[0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72],
               12:[0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85],
               13:[0, 2, 5, 25, 37, 43, 59, 70, 85, 89, 98, 99, 106],
               14:[0, 4, 6, 20, 35, 52, 59, 77, 78, 86, 89, 99, 122, 127],
               15:[0, 4, 20, 30, 57, 59, 62, 76, 100, 111, 123, 136, 144, 145, 151],
               16:[0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177],
               17:[0, 5, 7, 17, 52, 56, 67, 80, 81, 100, 122, 138, 159, 165, 168, 191, 199],
               18:[0, 2, 10, 22, 53, 56, 82, 83, 89, 98, 130, 148, 153, 167, 188, 192, 205, 216],
               19:[0, 1, 6, 25, 32, 72, 100, 108, 120, 130, 153, 169, 187, 190, 204, 231, 233, 242, 246],
               20:[0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179, 194, 208, 212, 228, 240, 253, 259, 283],
               21:[0, 2, 24, 56, 77, 82, 83, 95, 129, 144, 179, 186, 195, 255, 265, 285, 293, 296, 310, 329, 333],
               22:[0, 1, 9, 14, 43, 70, 106, 122, 124, 128, 159, 179, 204, 223, 253, 263, 270, 291, 330, 341, 353, 356],
               23:[0, 3, 7, 17, 61, 66, 91, 99, 114, 159, 171, 199, 200, 226, 235, 246, 277, 316, 329, 348, 350, 366, 372],
               24:[0, 9, 33, 37, 38, 97, 122, 129, 140, 142, 152, 191, 205, 208, 252, 278, 286, 326, 332, 353, 368, 384, 403, 425],
               25:[0, 12, 29, 39, 72, 91, 146, 157, 160, 161, 166, 191, 207, 214, 258, 290, 316, 354, 372, 394, 396, 431, 459, 467, 480	],
               26:[0, 1, 33, 83, 104, 110, 124, 163, 185, 200, 203, 249, 251, 258, 314, 318, 343, 356, 386, 430, 440, 456, 464, 475, 487, 492],
               27:[0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553]}


def get_basename(antenna_count=10, antenna_diameter=2.0, df=100e3, nf=200, fractional_spacing=1.0, f0=100e6):
    """Generate basename string for naming sim outputs and config yamls.

    Parameters
    ----------
    antenna_count: int, optional
        Number of antennas to simulate. Antennas will be arranged EW as a Golomb ruler.
        default is 10
    antenna_diameter: float, optional
        Diameter of antenna apertures (meters)
        default is 2.0
    df: float, optional
        frequency channel width (Hz)
        default is 100e3
    nf: integer, optional
        number of frequency channels to simulation
        Default is 200
    fractional_spacing: float, optional
        spacing between antennas as fraction of antenna_diameter
        Default is 1.0
    f0: float, optional
        minimum frequency to simulate (Hz)
        default is 100e6
    Returns
    -------
    basename: str
        basename string.
    """
    basename = f'HERA-III_antenna_diameter{antenna_diameter:.1f}_fractional_spacing{fractional_spacing:.1f}_nant{antenna_count}_nf{nf}_df{df/1e3:.3f}kHz_f0{f0/1e6:.3f}MHz'
    return basename


def initialize_telescope_yamls(output_dir='./', clobber=False, antenna_count=10, antenna_diameter=2.0, df=100e3, nf=200,
                              fractional_spacing=1.0, f0=100e6):
    """Initialize observing yaml files for simulation.

    Parameters
    ----------
    output_dir: str, optional
        directory to write simulation config files.
        default is current dir ('./')
    clobber: bool, optional
        overwrite existing config files.
        default is False
    antenna_count: int, optional
        Number of antennas to simulate. Antennas will be arranged EW as a Golomb ruler.
        default is 10
    antenna_diameter: float, optional
        Diameter of antenna apertures (meters)
        default is 2.0
    df: float, optional
        frequency channel width (Hz)
        default is 100e3
    nf: integer, optional
        number of frequency channels to simulation
        Default is 200
    fractional_spacing: float, optional
        spacing between antennas as fraction of antenna_diameter
        Default is 1.0
    f0: float, optional
        minimum frequency to simulate (Hz)
        default is 100e6
    Returns
    -------
    obs_param_yaml_name: str
        path to obs_param yaml file that can be fed into
        pyuvsim.simsetup.initialize_uvdata_from_params()
    telescope_yaml_name: str
        path to telescope yaml file. Referenced in obs_param_yaml file.
    csv_file: str
        path to csv file containing antenna locations.
    """
    basename = get_basename(antenna_count=antenna_count, antenna_diameter=antenna_diameter, df=df, nf=nf, fractional_spacing=fractional_spacing, f0=f0)
    antpos = np.asarray(golomb_dict[antenna_count]) * antenna_diameter * fractional_spacing
    csv_name =  os.path.join(output_dir, f'{basename}_antenna_layout.csv')
    telescope_yaml_name = os.path.join(output_dir, f'{basename}_telescope_config.yaml')

    telescope_yaml_dict = {'beam_paths': {i: {'type': 'airy'} for i in range(len(antpos))}, 'diameter': antenna_diameter,
                           'telescope_location': '(-30.721527777777847, 21.428305555555557, 1073.0000000046566)',
                           'telescope_name': 'HERA'}
    obs_param_dict = {'freq':{'Nfreqs': int(nf), 'bandwidth': float(nf * df), 'start_freq': float(f0)},
                      'telescope': {'array_layout': csv_name,
                                    'telescope_config_name': telescope_yaml_name},
                      'time': {'Ntimes': 1, 'duration_days': 0.0012731478148148147,
                               'integration_time': 11.0, 'start_time': 2457458.1738949567},
                      'polarization_array': [-5]}
    if not os.path.exists(telescope_yaml_name) or clobber:
        with open(telescope_yaml_name, 'w') as telescope_yaml_file:
            yaml.safe_dump(telescope_yaml_dict, telescope_yaml_file)
    # write csv file.
    lines = []
    lines.append('Name\tNumber\tBeamID\tE    \tN    \tU\n')
    for i, x  in enumerate(antpos):
        lines.append(f'ANT{i}\t{i}\t{i}\t{x:.4f}\t{0:.4f}\t{0:.4f}\n')
    if not os.path.exists(csv_name) or clobber:
        with open(csv_name, 'w') as csv_file:
            csv_file.writelines(lines)

    obs_param_yaml_name = os.path.join(output_dir, f'{basename}.yaml')
    if not os.path.exists(obs_param_yaml_name) or clobber:
        with open(obs_param_yaml_name, 'w') as obs_param_yaml:
            yaml.safe_dump(obs_param_dict, obs_param_yaml)
    return obs_param_yaml_name, telescope_yaml_name, csv_name


def initialize_simulation_uvdata(output_dir='./', clobber=False, keep_config_files_on_disk=False, compress_by_redundancy=False, **array_kwargs):
    """Prepare configuration files and UVData to run simulation.

    Parameters
    ----------
    output_dir: str, optional
        directory to write simulation config files.
    clobber: bool, optional
        overwrite existing config files.
    keep_config_files_on_disk: bool, optional
        Keep config files on disk. Otherwise delete.
        default is False.
    compress_by_redundancy: bool, optional
        If True, compress by redundancy. Makes incompatible with VisCPU for now.
    array_kwargs: kwarg_dict, optional
        array parameters. See get_basename for details.

    Returns
    -------
    uvd: UVData object
        blank uvdata file
    beam_ids: list
        list of beam ids
    beams: UVBeam list

    """
    # initialize telescope yaml
    obs_param_yaml_name, telescope_yaml_name, csv_name = initialize_telescope_yamls(output_dir=output_dir,
                                                                                    **array_kwargs)
    uvdata, beams, beam_ids = initialize_uvdata_from_params(obs_param_yaml_name)
    # remove obs config files.
    if not keep_config_files_on_disk:
        for yt in [obs_param_yaml_name, csv_name, telescope_yaml_name]:
            os.remove(yt)

    beam_ids = list(beam_ids.values())
    beams.set_obj_mode()
    _complete_uvdata(uvdata, inplace=True)
    if compress_by_redundancy:
        uvdata.compress_by_redundancy(tol = 0.25 * 3e8 / uvdata.freq_array.max(), inplace=True)
    return beam_ids, beams, uvdata


def run_simulation(eor_fg_ratio=1e-5, output_dir='./', nside_sky=256, clobber=False, compress_by_redundancy=True,
                   keep_config_files_on_disk=False, **array_config_kwargs):
    """End to end simulation of global sky-model with white noise EoR.

    Simulate visibilities at a single time for a Golomb array of antennas located at the HERA site.
    Uses the Global Sky Model (GSM) to compute foregrounds and simulates EoR signal as a white noise
    healpix map. Antenna configuration is saved to

    Parameters
    ----------
    eor_fg_ratio: float, optional
        ratio between stdev of eor and foregrounds over all healpix pixels.
        default is 1e-5
    output_dir: str, optional
        path to directory to output simulation products
        deault is './'
    nside_sky: int, optional
        healpix nside setting the resolution of the simulated sky.
        default is 256.
    clobber: bool, optional
        Overwrite existing UVData files.
        Default is False. If False, read any existing files and return them
        rather then simulating them.
    compress_by_redundancy: bool, optional
        If True, compress uvdata outputs by redundant averaging.
    output_dir: str, optional
        directory to write simulation config files.
    clobber: bool, optional
        overwrite existing config files.
    keep_config_files_on_disk: bool, optional
        Keep config files on disk. Otherwise delete.
        default is False.
    array_config_kwargs: kwarg_dict, optional
        array parameters. See initialize_simulation_uvdata for details.

    Returns
    -------
    uvd_gsm: UVData object
        UVData with visibilites of GSM emission.
    uvd_eor: UVData object
        UVData with visibilities of EoR emission.

    """
    # only perform simulation if clobber is true and gsm_file_name does not exist and eor_file_name does not exist:
    # generate GSM cube
    basename = get_basename(**array_config_kwargs)
    gsm_file_name = os.path.join(output_dir, basename + f'compressed_{compress_by_redundancy}_gsm.uvh5')
    eor_file_name = os.path.join(output_dir, basename + f'compressed_{compress_by_redundancy}_eor_{eor_fg_ratio:.1f}dB.uvh5')
    if not os.path.exists(gsm_file_name) or clobber:
        gsm = GlobalSkyModel(freq_unit='Hz')
        NPIX_GSM = hp.nside2npix(nside_sky)
        rot=hp.rotator.Rotator(coord=['G', 'C'])
        beam_ids, beams, uvdata = initialize_simulation_uvdata(output_dir=output_dir, clobber=clobber,
                                                               keep_config_files_on_disk=keep_config_files_on_disk,
                                                                **array_config_kwargs)
        gsmcube = np.random.randn(uvdata.Nfreqs, hp.nside2npix(nside_sky))
        for fnum, f in enumerate(uvdata.freq_array[0]):
            mapslice = gsm.generate(f)
            mapslice = hp.ud_grade(mapslice, nside_sky)
            # convert from galactic to celestial
            gsmcube[fnum] = rot.rotate_map_pixel(mapslice)
        # convert gsm cube from K to Jy / Sr. multiplying by 2 k_b / lambda^2 * ([Joules / meter^2 / Jy] =1e26)
        gsmcube = 2 * gsmcube * 1.4e-23 / 1e-26 / (3e8 / uvdata.freq_array[0][:, None])**2
        gsm_simulator = vis_cpu.VisCPU(uvdata=uvdata, sky_freqs=uvdata.freq_array[0], beams=beams,
                                       beam_ids=beam_ids, sky_intensity=gsmcube)
        gsm_simulator.simulate()
        gsm_simulator.uvdata.vis_units='Jy'
        uvd_gsm = gsm_simulator.uvdata
        if compress_by_redundancy:
            # compress with quarter wavelength tolerance.
            uvd_gsm.compress_by_redundancy(tol = 0.25 * 3e8 / uvd_gsm.freq_array.max())
        uvd_gsm.write_uvh5(gsm_file_name, clobber=True)
    else:
        uvd_gsm = UVData()
        uvd_gsm.read(gsm_file_name)
    # only do eor cube if file does not exist.
    if not os.path.exists(eor_file_name) or clobber:
        # initialize simulator
        beam_ids, beams, uvdata = initialize_simulation_uvdata(output_dir=output_dir, clobber=clobber,
                                                               keep_config_files_on_disk=keep_config_files_on_disk,
                                                               **array_config_kwargs)
        # define eor cube with random noise.
        eorcube = np.random.randn(uvdata.Nfreqs, hp.nside2npix(nside_sky))
        # make sure pixels >= zero.
        eorcube -= eorcube.min()
        eor_simulator = vis_cpu.VisCPU(uvdata=uvdata, sky_freqs=uvdata.freq_array[0],
                                       beams=beams, beam_ids=beam_ids, sky_intensity=eorcube)
        # simulator
        eor_simulator.simulate()
        # set visibility units.
        eor_simulator.uvdata.vis_units='Jy'
        # write out
        uvd_eor = eor_simulator.uvdata
        if compress_by_redundancy:
            # compress with quarter wavelength tolerance.
            uvd_eor.compress_by_redundancy(tol = 0.25 * 3e8 / uvd_eor.freq_array.max())
        uvd_eor.data_array *= np.sqrt(np.mean(np.abs(uvd_gsm.data_array) ** 2.))\
                              / np.sqrt(np.mean(np.abs(uvd_eor.data_array) ** 2.)) * eor_fg_ratio
        uvd_eor.write_uvh5(eor_file_name, clobber=True)
    else:
        # just read in if clobber=False and file already exists.
        uvd_eor = UVData()
        uvd_eor.read(eor_file_name)

    return uvd_gsm, uvd_eor


def simulate_and_filter(tol=1e-11, buffer_multiplier=1.0, antenna_diameter=2.0, output_dir='./', clobber=False, **sim_kwargs):
    """Simulate array configuration and apply filtering.

    Parameters
    ----------
    tol: float, optional
        amount to suppress foregrounds by in matrix inversion.
        default is 1e-11.
    buffer_multiplier: float, optional
        factor to multiply intrinsic chromaticity of dish by in determining filter width.
        default is 1.0
    output_dir: str, optional
        directory to store outputs in.
        default is './'
    clobber: bool, optional
        overwrite files if they exist.
    sim_kwargs: additional optional params for simulation.
        parameters of the sky simulation. See run_simulation docstring.
    """
    basename = get_basename(**sim_kwargs)
    filtered_pbl_file = os.path.join(output_dir, basename + f'_bmult{buffer_multiplier:.2f}_tol{tol:.1e}_pbl.uvh5')
    filtered_ibl_file = os.path.join(output_dir, basename + f'_bmult{buffer_multiplier:.2f}_tol{tol:.1e}_ibl.uvh5')
    if (not os.path.exists(filtered_pbl_file) or not os.path.exists(filtered_ibl_file)) or not skip_existing:
        uvd_eor_name = os.path.join(output_dir, basename + '_eor.uvh5')
        uvd_gsm_name = os.path.join(output_dir, basename + '_gsm.uvh5')
        if os.path.exists(uvd_eor_name) and os.path.exists(uvd_gsm_name):
            uvd_eor = UVData()
            uvd_eor.read_uvh5(uvd_eor_name)
            uvd_gsm = UVData()
            uvd_gsm.read_uvh5(uvd_gsm_name)
        else:
            uvd_eor, uvd_gsm = run_simulation(antenna_diameter=antenna_diameter, clobber=clobber, **sim_kwargs)
        uvd_total = copy.deepcopy(uvd_eor)
        uvd_total.data_array = uvd_eor.data_array + uvd_gsm.data_array
        uvd_filtered = filter_data(uvd_total, eta_max=antenna_diameter / 3e8 * buffer_multiplier, tol=tol, per_baseline=per_baseline, )
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
