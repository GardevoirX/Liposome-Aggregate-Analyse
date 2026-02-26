"""Tests for the PBC (Periodic Boundary Condition) Remover.

These tests verify that ``PBCRemover`` correctly translates atomic
coordinates so that aggregate atoms are contiguous in Cartesian space.
Test data lives in ``test/test_case/``.
"""

import MDAnalysis as mda
import numpy as np
import pytest
import torch
from MDAnalysis.analysis import distances
from rich.progress import track

from src.remove_pbc import PBCRemover


class TestPbcRemover:
    """Functional tests for ``PBCRemover``."""

    @pytest.fixture(scope="class")
    def pbc_setup(self):
        """Run PBCRemover on the small test-case and return both universes."""
        gro_file = "test/test_case/dry.gro"
        ndx_file = "test/test_case/agg_set"
        output_file = "test/test_case/test_output.gro"
        u = mda.Universe(gro_file, gro_file)
        remover = PBCRemover(gro_file, gro_file, ndx_file, output_file)
        remover.run()
        u_removed = mda.Universe(output_file, output_file)
        return u, u_removed, remover

    def test_move_to_unit_cell_dist_fidelity(self, pbc_setup):
        _u, _, remover = pbc_setup
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.tensor([0.5, 0.5, 0.7071])
        boxXYZ = torch.tensor([a[0], b[1], c[2]])
        dimensions = np.array([1.0, 1.0, 1.0, 60.0, 60.0, 90.0])
        position = torch.tensor([[0.0, 0.0, 0.0], boxXYZ / 2])

        for i in np.linspace(0.0, 10.0, 100):
            bias = torch.tensor([i, 0.0, 0.0])
            original_position = position
            changed_position = remover._move_to_unit_cell(
                original_position + bias, boxXYZ, a, b, c
            )
            self.dist_compare(
                2, original_position.numpy(), changed_position.numpy(), dimensions
            )

        for i in np.linspace(0.0, 10.0, 100):
            bias = torch.tensor([0.0, 0.0, i])
            original_position = position
            changed_position = remover._move_to_unit_cell(
                original_position + bias, boxXYZ, a, b, c
            )
            self.dist_compare(
                2, original_position.numpy(), changed_position.numpy(), dimensions
            )

    def test_atom_sequence(self, pbc_setup):
        u, u_removed, _ = pbc_setup
        assert len(u.atoms) == len(u_removed.atoms)
        for i_atom in range(len(u.atoms)):
            assert u.atoms[i_atom].name == u_removed.atoms[i_atom].name
            assert u.atoms[i_atom].resname == u_removed.atoms[i_atom].resname

    def test_atom_pair_dist_fidelity(self, pbc_setup):
        u, u_removed, _ = pbc_setup
        self.dist_compare(
            len(u.atoms), u.atoms.positions, u_removed.atoms.positions, u.dimensions
        )

    def dist_compare(self, n_atoms, pos1, pos2, box):

        for i_atom in track(
            range(min(n_atoms - 1, 4)), description="Testing atom pair distance..."
        ):
            original_dist = distances.distance_array(
                pos1[i_atom], pos1[i_atom + 1 :], box=box
            )
            output_dist = distances.distance_array(
                pos2[i_atom], pos2[i_atom + 1 :], box=box
            )

            assert np.sum(abs(original_dist - output_dist) > 1e-2) == 0
