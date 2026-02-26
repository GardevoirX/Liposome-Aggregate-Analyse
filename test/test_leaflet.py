"""Tests for the Leaflet data class (``src.leaflet``)."""

from __future__ import annotations

import numpy as np
import pytest

from src.leaflet import Leaflet


class TestLeafletCreation:
    """Verify ``Leaflet.__init__`` stores metadata correctly."""

    def test_basic_attributes(self):
        mol = np.array([0, 1, 2])
        lf = Leaflet(iFrame=0, iAgg=0, location=1, molIdx=mol)
        assert lf.iFrame == 0
        assert lf.iAgg == 0
        assert lf.location == 1
        assert lf.nMol == 3
        assert np.array_equal(lf.molIdx, mol)

    def test_single_molecule(self):
        lf = Leaflet(0, 0, 1, np.array([42]))
        assert lf.nMol == 1

    def test_empty_leaflet(self):
        lf = Leaflet(0, 0, 1, np.array([], dtype=int))
        assert lf.nMol == 0


class TestAddNewMol:
    """Verify ``Leaflet.add_new_mol``."""

    def test_adds_molecules(self):
        lf = Leaflet(0, 0, 1, np.array([1, 2, 3]))
        lf.add_new_mol(np.array([4, 5]))
        assert lf.nMol == 5
        assert np.array_equal(lf.molIdx, np.array([1, 2, 3, 4, 5]))

    def test_add_empty(self):
        lf = Leaflet(0, 0, 1, np.array([1, 2]))
        lf.add_new_mol(np.array([], dtype=int))
        assert lf.nMol == 2

    def test_add_to_empty(self):
        lf = Leaflet(0, 0, 1, np.array([], dtype=int))
        lf.add_new_mol(np.array([10, 20]))
        assert lf.nMol == 2


class TestGetComposition:
    """Verify ``Leaflet.get_composition``."""

    def test_basic_composition(self):
        mol_type = np.array(["POPC", "POPE", "POPC", "CHOL", "POPE"])
        lf = Leaflet(0, 0, 1, np.array([0, 2, 3]))
        comp = lf.get_composition(mol_type)
        assert comp["POPC"] == 2
        assert comp["CHOL"] == 1
        assert "POPE" not in comp or comp["POPE"] == 0

    def test_all_same_type(self):
        mol_type = np.array(["POPC", "POPC", "POPC"])
        lf = Leaflet(0, 0, 1, np.array([0, 1, 2]))
        comp = lf.get_composition(mol_type)
        assert comp["POPC"] == 3

    def test_composition_after_add(self):
        mol_type = np.array(["POPC", "POPE", "CHOL", "POPC"])
        lf = Leaflet(0, 0, 1, np.array([0]))
        lf.add_new_mol(np.array([2, 3]))
        comp = lf.get_composition(mol_type)
        assert comp["POPC"] == 2
        assert comp["CHOL"] == 1


class TestGetOutput:
    """Verify ``Leaflet.get_output`` prints correctly."""

    def test_output(self, capsys):
        lf = Leaflet(0, 0, 1, np.array([0, 1, 2]))
        lf.get_output()
        captured = capsys.readouterr()
        assert "1 2 3" in captured.out
