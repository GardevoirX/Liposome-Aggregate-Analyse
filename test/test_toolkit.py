"""Tests for toolkit utilities (``src.toolkit``)."""

from __future__ import annotations

import os

from src.toolkit import setEnv


class TestSetEnv:
    """Verify that ``setEnv`` configures thread-count environment variables."""

    def test_sets_all_thread_vars(self):
        setEnv(4)
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "4"
        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "4"
        assert os.environ["NUMEXPR_NUM_THREADS"] == "4"

    def test_single_thread(self):
        setEnv(1)
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_returns_zero(self):
        result = setEnv(8)
        assert result == 0
