"""Tests for the IO module (``src.io``)."""

import os
import tempfile

import numpy as np
import pytest

from src.io import read_file, write_file


class TestPickleRoundTrip:
    """Verify that ``write_file`` / ``read_file`` round-trip faithfully."""

    def test_dict_with_array(self, tmp_path):
        path = str(tmp_path / "data.pickle")
        data = {"arr": np.array([1.0, 2.0, 3.0]), "scalar": 42}
        write_file(data, path)
        loaded = read_file(path)
        assert np.array_equal(loaded["arr"], data["arr"])
        assert loaded["scalar"] == 42

    def test_plain_list(self, tmp_path):
        path = str(tmp_path / "list.pickle")
        data = [1, "two", 3.0, None]
        write_file(data, path)
        loaded = read_file(path)
        assert loaded == data

    def test_nested_structure(self, tmp_path):
        path = str(tmp_path / "nested.pickle")
        data = {
            "level1": {
                "level2": np.arange(10),
            },
            "flag": True,
        }
        write_file(data, path)
        loaded = read_file(path)
        assert np.array_equal(loaded["level1"]["level2"], data["level1"]["level2"])
        assert loaded["flag"] is True

    def test_empty_dict(self, tmp_path):
        path = str(tmp_path / "empty.pickle")
        write_file({}, path)
        loaded = read_file(path)
        assert loaded == {}

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_file("nonexistent_file_xyz.pickle")
