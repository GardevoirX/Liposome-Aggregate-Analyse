"""Tests for the Leaflet Analyzer module (``src.leaflet_analyzer``).

The ``Analyzer`` class loads leaflet assignment results from pickle files and
performs analyses such as leaflet location detection, molecule assignment, and
inter-leaflet correlation.

Test fixtures
-------------
Pickle paths are provided by session-scoped fixtures in ``conftest.py``.
By default these resolve to ``test/example/analyzer/``.  Pass
``--data-dir <path>`` to use an alternative fixture directory (e.g. the
full trajectory data on the Shanghai server).

Backward compatibility
----------------------
When ``vesicle_data.pickle`` is missing (or lacks certain keys), value-
comparison assertions are automatically *skipped* and only **smoke tests**
(the pipeline ran without error, return types are correct, etc.) execute.
This means the same test file works with both the 2-frame local example and
a full server-side dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.io import read_file
from src.leaflet_analyzer import Analyzer


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer(leaflet_pickle: str, args_pickle: str) -> Analyzer:
    """Shared ``Analyzer`` instance (created once per module)."""
    return Analyzer(leaflet_pickle, args_pickle)


@pytest.fixture(scope="module")
def vesicle_data(data_pickle: str | None) -> dict | None:
    """Expected assertion values loaded from ``vesicle_data.pickle``.

    Returns *None* when the file is absent so that value-comparison tests
    can skip gracefully.
    """
    if data_pickle is None:
        return None
    return read_file(data_pickle)


@pytest.fixture(scope="module")
def pipeline(analyzer: Analyzer) -> dict:
    """Run the full analysis pipeline on **frame 0** and cache results.

    Calling order inside the ``Analyzer`` matters — ``get_leaflet_location``
    must precede ``find_unassigned_molecules`` which must precede
    ``assign_molecules``.  This fixture guarantees the correct order and
    stores copies of the computed values so that individual test methods
    have no ordering dependency on each other.
    """
    analyzer.get_leaflet_location("vesicle", 0)
    unassigned = analyzer.find_unassigned_molecules(0)
    analyzer.assign_molecules(0)

    return {
        "unassigned": unassigned,
        "location_0_1": analyzer.leafletCollection[0][(0, 1)].location,
        "location_0_2": analyzer.leafletCollection[0][(0, 2)].location,
        "molIdx_1": analyzer.leafletCollection[0][(0, 1)].molIdx.copy(),
        "molIdx_2": analyzer.leafletCollection[0][(0, 2)].molIdx.copy(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expect(vesicle_data: dict | None, key: str):
    """Return ``vesicle_data[key]`` or *skip* when unavailable."""
    if vesicle_data is None:
        pytest.skip("No expected-values fixture (vesicle_data.pickle)")
    if key not in vesicle_data:
        pytest.skip(f"Key '{key}' not in vesicle_data (regenerate fixtures)")
    return vesicle_data[key]


# ===========================================================================
# Tests: Initialization (no pipeline needed)
# ===========================================================================


class TestAnalyzerInitialization:
    """Verify that ``Analyzer.__init__`` correctly loads data and computes
    derived attributes (residue counts, selected residues, head-atom
    indices, …).
    """

    # --- smoke tests (work with any data) ---

    def test_creates_instance(self, analyzer):
        assert analyzer is not None

    def test_total_res_num_positive(self, analyzer):
        assert analyzer.totalResNum > 0

    def test_not_solvent_bounded(self, analyzer):
        assert 0 < analyzer.notSolventResNum <= analyzer.totalResNum

    def test_selected_and_not_selected_partition(self, analyzer):
        """selectedRes + notSelectedRes should cover all residues."""
        all_res = set(analyzer.selectedRes) | set(analyzer.notSelectedRes)
        assert len(all_res) == analyzer.totalResNum

    def test_head_atom_idx_has_entries(self, analyzer):
        assert len(analyzer.headAtomIdx_noNan) > 0

    # --- value assertions (require vesicle_data.pickle) ---

    def test_total_res_num_value(self, analyzer, vesicle_data):
        expected = _expect(vesicle_data, "totalResNum")
        assert analyzer.totalResNum == expected

    def test_not_solvent_res_num_value(self, analyzer, vesicle_data):
        expected = _expect(vesicle_data, "notSolventResNum")
        assert analyzer.notSolventResNum == expected

    def test_selected_res(self, analyzer, vesicle_data):
        expected = _expect(vesicle_data, "selectedRes")
        assert (analyzer.selectedRes == expected).all()

    def test_not_selected_res(self, analyzer, vesicle_data):
        expected = _expect(vesicle_data, "notSelectedRes")
        assert (analyzer.notSelectedRes == expected).all()

    def test_head_atom_idx_no_nan(self, analyzer, vesicle_data):
        expected = _expect(vesicle_data, "headAtomIdx_noNan")
        assert (analyzer.headAtomIdx_noNan == expected).all()


# ===========================================================================
# Tests: Leaflet Location
# ===========================================================================


class TestLeafletLocation:
    """Verify leaflet location detection for vesicle geometry."""

    def test_leaflets_detected(self, analyzer, pipeline):
        assert (0, 1) in analyzer.leafletCollection[0]
        assert (0, 2) in analyzer.leafletCollection[0]

    def test_locations_positive(self, pipeline):
        assert pipeline["location_0_1"] > 0
        assert pipeline["location_0_2"] > 0

    def test_two_leaflets_differ(self, pipeline):
        assert not np.isclose(pipeline["location_0_1"], pipeline["location_0_2"])

    def test_location_values(self, pipeline, vesicle_data):
        expected_1 = _expect(vesicle_data, "location_0_1")
        expected_2 = _expect(vesicle_data, "location_0_2")
        assert np.isclose(pipeline["location_0_1"], expected_1)
        assert np.isclose(pipeline["location_0_2"], expected_2)


# ===========================================================================
# Tests: Molecule Assignment
# ===========================================================================


class TestMoleculeAssignment:
    """Verify unassigned-molecule detection and subsequent assignment."""

    def test_unassigned_is_ndarray(self, pipeline):
        assert isinstance(pipeline["unassigned"], np.ndarray)

    def test_assigned_covers_selected_res(self, analyzer, pipeline):
        """After assignment, every selected residue must be in a leaflet."""
        all_assigned = set(pipeline["molIdx_1"]) | set(pipeline["molIdx_2"])
        for res in analyzer.selectedRes:
            assert res in all_assigned

    def test_unassigned_values(self, pipeline, vesicle_data):
        expected = _expect(vesicle_data, "unAssigned")
        assert (pipeline["unassigned"] == expected).all()

    def test_assigned_idx_1(self, pipeline, vesicle_data):
        expected = _expect(vesicle_data, "assignedIdx1")
        assert (pipeline["molIdx_1"] == expected).all()

    def test_assigned_idx_2(self, pipeline, vesicle_data):
        expected = _expect(vesicle_data, "assignedIdx2")
        assert (pipeline["molIdx_2"] == expected).all()


# ===========================================================================
# Tests: Correlation
# ===========================================================================


class TestCorrelation:
    """Verify the inter-leaflet correlation calculation."""

    def test_calculate_correlation(self, analyzer, pipeline):  # noqa: ARG002
        """Correlation coefficient is a float in [-1, 1]."""
        atom_idx = analyzer.top.select_atoms("resname CHOL and name ROH").ids
        res_idx = analyzer.top.select_atoms("resname CHOL and name ROH").resids
        if len(atom_idx) == 0:
            pytest.skip("No CHOL residues in test data")
        corr = analyzer.calculate_correlation(0, 1, 50, atom_idx, res_idx)
        assert len(corr) == 1
        assert isinstance(corr[0], float)
        assert -1.0 <= corr[0] <= 1.0
