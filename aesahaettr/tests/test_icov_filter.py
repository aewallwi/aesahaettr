import numpy as np
from .. import icov_filter
import pytest



def test_filter_mat_simple(tmpdir):
    tmppath = tmpdir.strpath
    simple_filter_mat = icov_filter.filter_mat_simple(output_dir=tmppath, nf=13,
                                                      antenna_count=3, compress_by_redundancy=True)
    assert simple_filter_mat.shape[0] == 13 * (3 * 4 / 2 - 2)
