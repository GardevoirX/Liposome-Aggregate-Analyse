from src.remove_pbc import PBCRemover
import pytest
import torch
import numpy as np
import MDAnalysis as mda
from rich.progress import track
from MDAnalysis.analysis import distances

class Test_pbc_remover():

    @pytest.fixture()
    def test_running(self):

        groFile = 'test/test_case/dry.gro'
        ndxFile = 'test/test_case/agg_set'
        outputFile = 'test/test_case/test_output.gro'
        u = mda.Universe(groFile, groFile)
        remover = PBCRemover(groFile, groFile, ndxFile, outputFile)
        remover.run()
        uRemoved = mda.Universe(outputFile, outputFile)

        return u, uRemoved, remover

    def test_move_to_unit_cell_dist_fidelity(self, test_running):

        u, _, remover = test_running
        a = torch.tensor([1., 0., 0.])
        b = torch.tensor([0., 1., 0.])
        c = torch.tensor([0.5, 0.5, 0.7071])
        boxXYZ = torch.tensor([a[0], b[1], c[2]])
        dimensions = np.array([1., 1., 1., 60., 60., 90.])
        position = torch.tensor([[0., 0., 0.],
                                 boxXYZ / 2])
        
        for i in np.linspace(0., 10., 100):
            bias = torch.tensor([i, 0., 0.])
            original_position = position
            changed_position = remover._move_to_unit_cell(
                    original_position + bias, boxXYZ, a, b, c)
            self.dist_compare(
                    2, original_position.numpy(), 
                    changed_position.numpy(), dimensions)
            
        for i in np.linspace(0., 10., 100):
            bias = torch.tensor([0., 0., i])
            original_position = position
            changed_position = remover._move_to_unit_cell(
                    original_position + bias, boxXYZ, a, b, c)
            self.dist_compare(
                    2, original_position.numpy(), 
                    changed_position.numpy(), dimensions)

    def test_atom_sequence(self, test_running):

        u, uRemoved, _ = test_running
        assert len(u.atoms) == len(uRemoved.atoms)
        for iAtom in range(len(u.atoms)):
            assert u.atoms[iAtom].name == uRemoved.atoms[iAtom].name
            assert u.atoms[iAtom].resname == uRemoved.atoms[iAtom].resname

    def test_atom_pair_dist_fidelity(self, test_running):

        u, uRemoved, _ = test_running
        self.dist_compare(len(u.atoms), u.atoms.positions, uRemoved.atoms.positions, u.dimensions)
    
    def dist_compare(self, nAtoms, pos1, pos2, box):

        for iAtom in track(range(min(nAtoms - 1, 4)), description='Testing atom pair distance...'):
            originalDist = distances.distance_array(
                    pos1[iAtom], pos1[iAtom + 1:], box=box)
            outputDist = distances.distance_array(
                    pos2[iAtom], pos2[iAtom + 1:], box=box)

            assert np.sum(abs(originalDist - outputDist) > 1e-2) == 0
                

