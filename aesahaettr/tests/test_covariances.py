from .. import covariances
import pytest
import numpy as np

def test_cov_mat_simple(tmpdir):
    tmppath = tmpdir.strpath
    for blcut in [np.inf, 10., 100.]:
        for obbl in [True, False]:
            cov_mat = covariances.cov_mat_simple(antenna_count=4, nf=11,
                                                 output_dir=tmppath, order_by_bl_length=obbl)
            assert cov_mat.shape[0] == 4 * 5 / 2 * 11
            assert cov_mat.shape[1] == cov_mat.shape[0]
            assert cov_mat.ndim == 2

def test_cov_mat_simple_evecs(tmpdir):
    tmppath = tmpdir.strpath
    for blcut in [np.inf, 0., 10]:
        for compress in [True, False]:
            evecs = {}
            evals={}
            for sparse in [True, False]:
                evals[sparse], evecs[sparse] = covariances.cov_mat_simple_evecs(use_sparseness=sparse, output_dir=tmppath, antenna_count=4, nf=11,
                                                                                bl_cutoff_buffer=blcut, fractional_spacing=1.23,
                                                                                compress_by_redundancy=compress)
                if not compress:
                    assert evecs[sparse].shape[1] == 4 * 5 / 2
                else:
                    assert evecs[sparse].shape[1] == (4 * 5 / 2 - 3)
                assert evecs[sparse].shape[2] == 11
            # test that sparse and non sparse are close.
            if blcut == 0.:
                nvals = min(len(evals[True]), len(evals[False]))
                assert np.allclose(evals[True][:nvals], evals[False][:nvals])
                for j in range(nvals):
                    assert np.allclose(np.abs(evecs[True][j]), np.abs(evecs[False][j]), rtol=0., atol=1e-6)



def test_cov_mat_simulated(tmpdir):
    tmppath = tmpdir.strpath
    for compress in [True, False]:
        for mode in ['gsm', 'eor']:
            for order_by_bl_length in [True, False]:
                cmat_sim = covariances.cov_mat_simulated(ndraws=10, compress_by_redundancy=compress, nf=13,
                                                         mode=mode, order_by_bl_length=order_by_bl_length, antenna_count=4,
                                                         fractional_spacing=1.23, output_dir=tmppath, nside_sky=8)
                if compress:
                    assert cmat_sim.shape[0] == (13 * (4 * 5 / 2 - 3))
                    assert cmat_sim.shape[1] == cmat_sim.shape[0]
                else:
                    assert cmat_sim.shape[0] == (13 * 4 * 5 / 2)
                    assert cmat_sim.shape[1] == cmat_sim.shape[0]

def test_cov_matrix_airy(tmpdir):
    tmppath = tmpdir.strpath
    for compress in [True, False]:
        for corr_freqs in [True, False]:
            for order_by_bl_length in [True, False]:
                cmat_airy = covariances.cov_matrix_airy(compress_by_redundancy=compress, nf=13,
                                                        correlated_freqs=corr_freqs,
                                                        order_by_bl_length=order_by_bl_length, antenna_count=4,
                                                        fractional_spacing=1.23, output_dir=tmppath)
                if compress:
                    assert cmat_airy.shape[0] == (13 * (4 * 5 / 2 - 3))
                    assert cmat_airy.shape[1] == cmat_airy.shape[0]
                else:
                    assert cmat_airy.shape[0] == (13 * 4 * 5 / 2)
                    assert cmat_airy.shape[1] == cmat_airy.shape[0]
