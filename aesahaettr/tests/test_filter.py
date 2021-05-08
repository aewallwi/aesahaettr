import numpy as np
from . import filter
import pytest
import simulator



def test_filter_mat_simple(tmpdir):
    tmppath = tmpdir.strpath
