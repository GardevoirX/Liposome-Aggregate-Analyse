"""Tests for selection parsing utilities (``src.selection``)."""

import pytest

from src.selection import get_atom_selection


class TestGetAtomSelection:
    """Verify the ``resname:atom1 atom2|...`` DSL parser."""

    def test_single_residue_single_atom(self):
        result = get_atom_selection("POPC:GL1")
        assert result == {"POPC": ["GL1"]}

    def test_single_residue_multiple_atoms(self):
        result = get_atom_selection("POPC:GL1 GL2")
        assert result == {"POPC": ["GL1", "GL2"]}

    def test_multiple_residues(self):
        result = get_atom_selection("APC:GL2|DOPC:GL1 GL2|DOPE:GL1 GL2")
        assert result == {
            "APC": ["GL2"],
            "DOPC": ["GL1", "GL2"],
            "DOPE": ["GL1", "GL2"],
        }

    def test_sphingomyelin_am_atoms(self):
        """Sphingomyelins use AM1 AM2 head atoms."""
        result = get_atom_selection("DPSM:AM1 AM2|BNSM:AM1 AM2")
        assert result == {
            "DPSM": ["AM1", "AM2"],
            "BNSM": ["AM1", "AM2"],
        }

    def test_preserves_order(self):
        result = get_atom_selection("X:C B A")
        assert result["X"] == ["C", "B", "A"]

    def test_many_residues_count(self):
        """A realistic head-atom selection should parse all residue types."""
        sel = (
            "APC:GL2|BNSM:AM1 AM2|DAPC:GL1 GL2|DAPE:GL1 GL2|DAPS:GL1 GL2|"
            "DOPC:GL1 GL2|DOPE:GL1 GL2|POPC:GL1 GL2|POPE:GL1 GL2"
        )
        result = get_atom_selection(sel)
        assert len(result) == 9
        assert "APC" in result
        assert "POPE" in result
