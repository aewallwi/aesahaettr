import numpy as np
import itertools
import scipy.special as sp
import copy
import healpy as hp
import os
import yaml
from pyuvdata import UVData
from hera_sim.visibilities import vis_cpu
from pyuvsim.simsetup import initialize_uvdata_from_params, _complete_uvdata
from . import defaults

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


def get_basename(antenna_count=defaults.antenna_count, antenna_diameter=defaults.antenna_diameter,
                 df=defaults.df, nf=defaults.nf, fractional_spacing=defaults.fractional_spacing,
                 f0=defaults.f0):
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


def initialize_telescope_yamls(output_dir='./', clobber=False, antenna_count=defaults.antenna_count, antenna_diameter=defaults.antenna_diameter,
                               df=defaults.df, nf=defaults.nf, fractional_spacing=defaults.fractional_spacing, f0=defaults.f0):
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
    telescope_yaml_name = os.path.join(output_dir, f'{basename}_telescope_defaults.yaml')

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

def initialize_uvdata(output_dir='./', clobber=False, keep_config_files_on_disk=False, compress_by_redundancy=False,
                      **array_kwargs):
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
    beams: UVBeam list
    beam_ids: list
        list of beam ids

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
    return uvdata, beams, beam_ids


def compute_visibilities(eor_fg_ratio=1e-5, output_dir='./', nside_sky=defaults.nside_sky, clobber=False, compress_by_redundancy=True,
                         keep_config_files_on_disk=False, include_autos=False, include_gsm=True, include_gleam=True, nsrcs_gleam=10000, **array_config_kwargs):
    """Compute visibilities for global sky-model with white noise EoR.

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
    include_gsm: bool, optional.
        include desourced gsm in sky model.
        default is True.
    include_gleam: bool, optional.
        include gleam point sources in sky model.
        default is True.
    nsrcs_gleam: int, optional
        number of brightest gleam sources to include in sky model
        default is 10000
    array_config_kwargs: kwarg_dict, optional
        array parameters. See initialize_uvdata for details.

    Returns
    -------
    uvd_fg: UVData object
        UVData with visibilites of foreground emission.
    uvd_eor: UVData object
        UVData with visibilities of EoR emission.

    """
    # only perform simulation if clobber is true and fg_file_name does not exist and eor_file_name does not exist:
    # generate GSM cube
    basename = get_basename(**array_config_kwargs)
    fg_file_name = os.path.join(output_dir, basename + f'compressed_{compress_by_redundancy}_autos{include_autos}_fg_{include_gsm}_gleam_{include_gleam}_nsrc_{nsrcs_gleam}.uvh5')
    eor_file_name = os.path.join(output_dir, basename + f'compressed_{compress_by_redundancy}_autos{include_autos}_eor_{np.log10(eor_fg_ratio) * 10:.1f}dB.uvh5')
    if not os.path.exists(fg_file_name) or clobber:
        from . import skymodel
        uvdata, beams, beam_ids = initialize_uvdata(output_dir=output_dir, clobber=clobber,
                                                               keep_config_files_on_disk=keep_config_files_on_disk,
                                                                **array_config_kwargs)
        if include_gsm:
            fgcube = skymodel.initialize_gsm(uvdata.freq_array[0], nside_sky=nside_sky, output_dir=output_dir)
        else:
            fgcube = np.zeros((len(uvdata.freq_array[0], hp.nside2npix(nside_sky))))
        if include_gleam:
            fgcube = skymodel.add_gleam(uvdata.freq_array[0], fgcube, nsrcs=nsrcs_gleam)
        fg_simulator = vis_cpu.VisCPU(uvdata=uvdata, sky_freqs=uvdata.freq_array[0], beams=beams,
                                       beam_ids=beam_ids, sky_intensity=fgcube)
        fg_simulator.simulate()
        fg_simulator.uvdata.vis_units='Jy'
        uvd_fg = fg_simulator.uvdata
        if compress_by_redundancy:
            # compress with quarter wavelength tolerance.
            uvd_fg.compress_by_redundancy(tol = 0.25 * 3e8 / uvd_fg.freq_array.max())
        if not include_autos:
            uvd_fg.select(bls=[ap for ap in uvd_fg.get_antpairs() if ap[0] != ap[1]], inplace=True)
        uvd_fg.write_uvh5(fg_file_name, clobber=True)
    else:
        uvd_fg = UVData()
        uvd_fg.read(fg_file_name)
    # only do eor cube if file does not exist.
    if not os.path.exists(eor_file_name) or clobber:
        from . import skymodel
        # initialize simulator
        uvdata, beams, beam_ids = initialize_uvdata(output_dir=output_dir, clobber=clobber,
                                                               keep_config_files_on_disk=keep_config_files_on_disk,
                                                               **array_config_kwargs)
        # define eor cube with random noise.
        eorcube = skymodel.initialize_eor(uvdata.freq_array[0], nside_sky)
        # make sure pixels >= zero.
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
        if not include_autos:
            uvd_eor.select(bls=[ap for ap in uvd_eor.get_antpairs() if ap[0] != ap[1]], inplace=True)
        uvd_eor.data_array *= np.sqrt(np.mean(np.abs(uvd_fg.data_array) ** 2.))\
                              / np.sqrt(np.mean(np.abs(uvd_eor.data_array) ** 2.)) * eor_fg_ratio
        uvd_eor.write_uvh5(eor_file_name, clobber=True)
    else:
        # just read in if clobber=False and file already exists.
        uvd_eor = UVData()
        uvd_eor.read(eor_file_name)

    return uvd_fg, uvd_eor

def grid_nearest_neighbor(uvdata=None, **array_kwargs):
    """Perform nearest neighbor gridding of visibilities to a u-nu plane.
    """
    return

def get_sampling(uvdata=None, **array_kwargs):
    """get sampling of the u-frequency plane for a 1d array.

    Parameters
    ----------
    ax: matplotlib axis handle
        axis to draw plot on. If None provided, create new axis.
    uvdata: UVData object
        UVdata object to visualize u-eta sampling for.
        If None is provided, generate a UVData object from array_kwargs
        and defaults.py parameters.

    Returns
    -------
    ax, axis handle of current plot.
    u_axis: u axis of sampling grid.
    nu_axis: frequency axis of sampling grid.
    sampling: array-like
        2d array of u-nu sampling
    sampling_collapsed:
        u-only sampling
    """
    if 'antenna_diameter' in array_kwargs:
        antenna_diameter = array_kwargs['antenna_diameter']
    else:
        antenna_diameter = defaults.antenna_diameter
    if uvdata is None:
        uvdata, _, _ = initialize_uvdata(**array_kwargs)
    umax = uvdata.uvw_array[:, 0].max() * uvdata.freq_array.max() / 3e8
    umin = uvdata.uvw_array[:, 0].min() * uvdata.freq_array.max() / 3e8
    # half wave sampling.
    uaxis = np.linspace(umin, umax, int(2 * np.ceil(umax - umin)))
    nuaxis = uvdata.freq_array[0]
    sampling = np.zeros((len(uaxis), len(nuaxis)))
    times = np.unique(uvdata.time_array)
    for time in times:
        time_indices = np.where(uvdata.time_array==time)[0]
        for fnum, freq in enumerate(uvdata.freq_array[0]):
            for tind in time_indices:
                u_overlap = np.where(np.abs(uvdata.uvw_array[tind, 0] * freq / 3e8 - uaxis) <= antenna_diameter * freq / 3e8)
                sampling[u_overlap, fnum] += 1.0
    sampling_collapsed = np.sum(sampling, axis=1)
    return uaxis, nuaxis, sampling, sampling_collapsed
