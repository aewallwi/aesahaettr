import numpy as np
from .. import icov_filter
import pytest



def test_filter_mat_simple(tmpdir):
    tmppath = tmpdir.strpath
    for sparse in [True, False]:
        for blcut in [0., 10., np.inf]:
            simple_filter_mat = icov_filter.filter_mat_simple(output_dir=tmppath, nf=13, bl_cutoff_buffer=blcut,
                                                              antenna_count=3, compress_by_redundancy=True, use_sparseness=sparse)
    assert simple_filter_mat.shape[0] == 13 * (3 * 4 / 2 - 2)

def test_filter_covariance():
    pytest.raises(NotImplementedError, icov_filter.filter_covariance, np.random.randn(10, 10))
