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
