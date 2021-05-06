import numpy as np
import itertools
import scipy.special as sp
import copy
import healpy as hp
from pyuvsim import analyticbeam as ab
import os
from aesahaettr import filter
import tqdm
import yaml
from pygsm import GlobalSkyModel


golomb_dict = {0:[0], 1:[0,1], 2:[0,1,3],
               3:[0,1,4,6], 4:[0,1,4,6], 5:[0,1,4,9,11],
               6:[0,1,4,10,12,17], 7:[0,1,4,10,18,23,25],
               8:[0,1,4,9,15,22,32,34], 9:[0,1,5,12,25,27,35,41,44],
               10:[0,1,6,10,23,26,34,41,53,55],
               11:[0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72]
               12:[0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85]
               13:[0, 2, 5, 25, 37, 43, 59, 70, 85, 89, 98, 99, 106]
               14:[0, 4, 6, 20, 35, 52, 59, 77, 78, 86, 89, 99, 122, 127]
               15:[0, 4, 20, 30, 57, 59, 62, 76, 100, 111, 123, 136, 144, 145, 151],
               16:[0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177],
               17:[0, 5, 7, 17, 52, 56, 67, 80, 81, 100, 122, 138, 159, 165, 168, 191, 199],
               18:[0, 2, 10, 22, 53, 56, 82, 83, 89, 98, 130, 148, 153, 167, 188, 192, 205, 216],
               19:[0, 1, 6, 25, 32, 72, 100, 108, 120, 130, 153, 169, 187, 190, 204, 231, 233, 242, 246],
               20:[0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179, 194, 208, 212, 228, 240, 253, 259, 283],
               21:[0, 2, 24, 56, 77, 82, 83, 95, 129, 144, 179, 186, 195, 255, 265, 285, 293, 296, 310, 329, 333]
               22:[0, 1, 9, 14, 43, 70, 106, 122, 124, 128, 159, 179, 204, 223, 253, 263, 270, 291, 330, 341, 353, 356],
               23:[0, 3, 7, 17, 61, 66, 91, 99, 114, 159, 171, 199, 200, 226, 235, 246, 277, 316, 329, 348, 350, 366, 372],
               24:[0, 9, 33, 37, 38, 97, 122, 129, 140, 142, 152, 191, 205, 208, 252, 278, 286, 326, 332, 353, 368, 384, 403, 425],
               25:[0, 12, 29, 39, 72, 91, 146, 157, 160, 161, 166, 191, 207, 214, 258, 290, 316, 354, 372, 394, 396, 431, 459, 467, 480	],
               26:[0, 1, 33, 83, 104, 110, 124, 163, 185, 200, 203, 249, 251, 258, 314, 318, 343, 356, 386, 430, 440, 456, 464, 475, 487, 492],
               27:[0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553]}

def get_basename(antenna_count, dish_size, df, nf, fractional_spacing, f0):
        return f'HERA-III_dish_size{dish_size}_fractional_spacing{fractional_spacing}_G{antenna_count}_nf{nf}_df{df/1e3:.3f}kHz_f0{f0/1e6:.3f}'

def run_simulation(antenna_count=10, dish_size=2.0, df=100e3, nf=200, fractional_spacing=1., dynamic_range_db=-5.5, f0=100e6, output_dir='./', NSIDE_GSM=256):

    golomb_dict = {k:np.asarray(golomb_dict[k]) for k in golomb_dict}
    antpos = golomb_dict[antenna_count] * dish_size * fractional_spacing
    obs_freqs = np.linspace(f0, f0 + nf * df, nf)
    # write yaml file
    telescope_yaml_dict = {'beam_paths': {i: {'type': 'airy'} for i in range(len(antpos))}, 'diameter': dish_size,
                           'telescope_location': '(-30.721527777777847, 21.428305555555557, 1073.0000000046566)',
                           'telescope_name': 'HERA'}
    telescope_yaml_name = os.path.join(output_dir, f'airy_config_dish_size{dish_size}.yaml')
    with open(telescope_yaml_name, 'w') as telescope_yaml_file:
        yaml.safe_dump(telescope_yaml_dict, telescope_yaml_file)
    # write csv file.
    csv_name =  os.path.join(output_dir, f'layout_dish_size{dish_size}_fractional_spacing{fractional_spacing}_G{antenna_count}.csv')
    lines = []
    lines.append('Name\tNumber\tBeamID\tE    \tN    \tU\n')
    for i, x  in enumerate(antpos):
        lines.append(f'ANT{i}\t{i}\t{i}\t{x:.4f}\t{0:.4f}\t{0:.4f}\n')
    with open(csv_name, 'w') as csv_file:
        csv_file.writelines(lines)
    # generate obs param dict and csv file
    obs_param_dict = {'freq':{'Nfreqs': int(nf), 'bandwidth': float(nf * df), 'start_freq': float(obs_freqs[0])},
                      'telescope': {'array_layout': csv_name,
                                    'telescope_config_name': telescope_yaml_name},
                      'time': {'Ntimes': 1, 'duration_days': 0.0012731478148148147,
                               'integration_time': 11.0, 'start_time': 2457458.1738949567},
                      'polarization_array': [-5]}
    basename = get_basename(antenna_count, dish_size, df, nf, fractional_spacing, f0)
    obs_param_yaml_name = os.path.join(output_dir, f'{basename}.yaml')
    with open(obs_param_yaml_name, 'w') as obs_param_yaml:
        yaml.safe_dump(obs_param_dict, obs_param_yaml)
    # generate GSM cube
    gsm = GlobalSkyModel(freq_unit='Hz')
    NPIX_GSM = hp.nside2npix(NSIDE_GSM)
    gsmcube = np.zeros((nf, NPIX_GSM))
    rot=hp.rotator.Rotator(coord=['G', 'C'])
    for fnum,f in enumerate(obs_freqs):
        mapslice = gsm.generate(f)
        mapslice = hp.ud_grade(mapslice, NSIDE_GSM)
        # convert from galactic to celestial
        gsmcube[fnum] = rot.rotate_map_pixel(mapslice)
    gsmcube = np.array(gsmcube)
    gsmcube = 2 * gsmcube * np.pi * 4 / NPIX_GSM * 1.4e-23 / 1e-26 / (3e8 / obs_freqs[:, None])**2\
            / hp.nside2pixarea(NSIDE_GSM)
    from hera_sim.visibilities import vis_cpu
    # define eor cube with random noise.
    eorcube = np.abs(np.random.randn(*gsmcube.shape))
    eorcube *= np.std(gsmcube) / np.std(eorcube) * 10. ** (dynamic_range_db)
    from pyuvsim.simsetup import initialize_uvdata_from_params, _complete_uvdata
    uvdata, beams, beam_ids = initialize_uvdata_from_params(obs_param_yaml_name)
    _complete_uvdata(uvdata, inplace=True)
    beam_ids = list(beam_ids.values())
    beams.set_obj_mode()
    eor_simulator = vis_cpu.VisCPU(uvdata=copy.deepcopy(uvdata), sky_freqs=obs_freqs, beams=beams, beam_ids=beam_ids, sky_intensity=eorcube)
    eor_vis = eor_simulator.simulate()
    gsm_simulator = vis_cpu.VisCPU(uvdata=copy.deepcopy(uvdata), sky_freqs=obs_freqs, beams=beams, beam_ids=beam_ids, sky_intensity=gsmcube)
    gsm_vis = gsm_simulator.simulate()
    # set visibility units.
    eor_simulator.uvdata.vis_units='Jy'
    eor_simulator.uvdata.vis_units='Jy'
    # write uvdata files.
    gsm_simulator.uvdata.write_uvh5(basename + '_gsm.uvh5', clobber=True)
    eor_simulator.uvdata.write_uvh5(basename + f'_eor_{dynamic_range_db:.1f}dB.uvh5', clobber=True)
    return gsm_simulator.uvdata, eor_simulator.uvdata


def process_configuration(antenna_count=10, dish_size=2.0, df=100e3, nf=200, fractional_spacing=1.,
                          dynamic_range_db=-5.5, tol=1e-11, f0=100e6, output_dir='./',
                          buffer_multiplier=1.0, skip_existing=True):
    basename = get_basename(antenna_count, dish_size, df, nf, fractional_spacing, f0)
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
            uvd_eor, uvd_gsm = run_simulation(antenna_count=antenna_count,
                                              dish_size=dish_size, df=df, nf=nf, fractional_spacing=fractional_spacing,
                                              dynamic_range_db=dynamic_range_db, output_dir=output_dir)
        uvd_total = copy.deepcopy(uvd_eor)
        uvd_total.data_array = uvd_eor.data_array + uvd_gsm.data_array
        uvd_filtered_pbl = filter_data(uvd_total, eta_max=2 / 3e8 * buffer_multiplier, tol=tol)
        uvd_filtered_ibl = filter_data(uvd_total, eta_max=2 / 3e8 * buffer_multiplier, tol=tol, per_baseline=False)
        uvd_filtered_pbl.write_uvh5(filtered_pbl_file, clobber=True)
        uvd_filtered_ibl.write_uvh5(filtered_ibl_file, clobber=True)


def get_simulation_parser():
    ap = argparse.ArgumentParser(description="Perform a 1-dimensional filtering simulation.")
    ap.add_argument("--antenna_count", type=int, help="Number of antennas in array simulation.", default=10)
    ap.add_argument("--max_bl_length", type=int, help="Maximum baseline length. Can be used to automatically determine the antenna count.", default=None)
    ap.add_argument("--dish_diameter", type=float, default=4.0, help="Diameter of a single antenna element (uses Airy beam).")
    ap.add_argument("--fractional_spacing", type=float, help="Distance between elements as a fraction of dish diameter.", default=1.0)
    ap.add_argument("--Nfreqs", type=int, help="Number of frequency channels.", default=100)
    ap.add_argument("--df", type=float, help="Frequency Channel width.", default=100e3)
    ap.add_argument("--")





def generate_grid_params(r_spaces=None, dish_sizes=None, nfs=None, antenna_counts=None, dfs=None):

    r_spaces = [1, 1.5, 2.]
    dish_sizes = [2.0, 4.0, 8.0, 14.0]
    nfs = [100, 200, 400]
    antenna_counts = [10, 15]
    dfs = [100e3, 200e3]
    param_combinations = []

    for fractional_spacing in r_spaces:
        for dish_size in dish_sizes:
            for nf in nfs:
                for df in dfs:
                    for antenna_count in antenna_counts:
                        param_combinations.append({'fractional_spacing':fractional_spacing, 'dish_size':dish_size, 'nf':nf, 'df':df, 'antenna_count':antenna_count})
