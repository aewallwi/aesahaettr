import numpy as np
from .. import filter
import pytest



def test_filter_mat_simple(tmpdir):
    tmppath = tmpdir.strpath
    simple_filter_mat = filter.filter_mat_simple(output_dir=tmppath, nf=13,
                                                 antenna_count=3, compress_by_redundancy=True)
    assert simple_filter_mat.shape[0] == 13 * (3 * 4 / 2 - 2)


def test_filter_data(tmpdir):
    tmppath = tmpdir.strpath
    
