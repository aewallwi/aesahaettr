from .. import calamity
from .. import covariances
from .. import visibilities
import copy
import numpy as np


def test_fit_foregrounds(tmpdir):
    # test fitting foregrounds
    uvd_gsm, uvd_eor = visibilities.compute_visibilities(
        output_dir=tmpdir,
        nside_sky=8,
        antenna_count=5,
        nf=10,
        df=1e6,
        f0=100e6,
        antenna_diameter=2.0,
        fractional_spacing=5,
    )
    uvd_total = copy.deepcopy(uvd_gsm)
    uvd_total.data_array += uvd_eor.data_array

    evals, evecs = covariances.cov_mat_simple_evecs(uvdata=uvd_total, antenna_diameter=2.0)
    # initialize with true foreground coefficients
    x0 = evecs.T @ (uvd_gsm.data_array.reshape(uvd_gsm.Nbls * uvd_gsm.Nfreqs))
    # plus a perturbation.
    foreground_coefficients = calamity.fit_foregrounds(
        uvd_total.data_array.squeeze().reshape(uvd_total.Nbls * uvd_total.Nfreqs),
        foreground_coefficients=x0,
        foreground_basis_vectors=evecs,
    )
    foreground_coefficients += 0.3 * foreground_coefficients * (np.random.randn(len(foreground_coefficients)))
    foreground_model = (evecs @ foreground_coefficients).reshape(uvd_gsm.Nbls, uvd_gsm.Nfreqs)
    assert np.allclose(foreground_model[:, None, :, None], uvd_gsm.data_array)
