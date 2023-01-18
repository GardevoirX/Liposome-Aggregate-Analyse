import pickle
import warnings
import numpy as np
import mdtraj as md
from src.define import *
from src.toolkit import read_file, write_file
from src.selection import *

class Analyzer():

    def __init__(self, fileName, argsFileName):

        self.leafletCollection = read_file(fileName)
        self.args = read_file(argsFileName)
        self.trajFileName = self.args.trajFile
        self.top = md.load(self.args.topFile)
        self.headAtomSelection = fix_not_selected_mol(self.top, \
                get_atom_selection(self.args.selHeadAtom))
        self.headAtomIdx = get_index_of_selected_atom(self.top, \
                self.headAtomSelection)
        self.selectedRes = np.arange(self.top.top.n_residues)[
                ~np.isnan(self.headAtomIdx)]
        self.notSelectedRes = np.arange(self.top.top.n_residues)[
                np.isnan(self.headAtomIdx)]
        self.headAtomIdx_noNan = np.array(
                self.headAtomIdx[self.selectedRes], dtype=int)
        self.unassignedMolIdx = {}
        self.leafletLocation = {}

    def load_traj(self, stride=None):

        self.traj = md.load(self.trajFileName, top=self.top, stride=stride)

        return self.traj

    def get_leaflet_location(self, iFrame, leafletType):

        if leafletType == 'membrane':
            if len(self.leafletCollection[iFrame]) \
                    != 2:
                warnings.warn("Warning! The number of leaflet in your aggregate is not 2!")
            atomIdx1 = np.array(
                    self.headAtomIdx[
                        self.leafletCollection[0][(0, 1)].molIdx], dtype=int)
            atomIdx2 = np.array(
                    self.headAtomIdx[
                        self.leafletCollection[0][(0, 2)].molIdx], dtype=int)
            leafletLocation = (
                np.average(self.traj.xyz[iFrame][atomIdx1][::, -1]),
                np.average(self.traj.xyz[iFrame][atomIdx2][::, -1])
            )
        
        self.leafletLocation[iFrame] = leafletLocation

        return leafletLocation

    def get_aggregate_molecules(self):

        self.aggAtomIdx = self.top.top.select('not resname ' + ' '.join(RESEXCLUDED))
        self.aggResIdx = [iRes for iRes, res in enumerate(self.top.top.residues) if not res.name in MOLEXCLUDED]

    def find_unassigned_molecules(self, iFrame):

        assignedMolIdx = np.concatenate(
            [self.leafletCollection[iFrame][iter].molIdx 
                for iter in self.leafletCollection[iFrame]])
        self.unassignedMolIdx[iFrame] = []
        for res in self.top.top.residues:
            if (res.index not in assignedMolIdx) \
                    and res.name not in MOLEXCLUDED:
                self.unassignedMolIdx[iFrame].append(res.index)
        self.unassignedMolIdx[iFrame] = np.array(
                self.unassignedMolIdx[iFrame])

        return self.unassignedMolIdx[iFrame]


    def assign_molecules(self, iFrame, leafletType):

        # TODO: assign molecules according to its z coordinate or the distance to vesicle center
        headAtomCoord = []
        headAtomIdx = []
        if leafletType == 'membrane':
            for iRes in self.unassignedMolIdx[iFrame]:
                headAtomIdx.append(self.top.top.residue(iRes).atom(0).index)
                #select the head bead of the molecule
            headAtomCoord = self.traj.xyz[iFrame][headAtomIdx][::, -1]
            inLeaflet1 = abs(headAtomCoord - self.leafletLocation[iFrame][0])\
                    < abs(headAtomCoord - self.leafletLocation[iFrame][1])
            self.leafletCollection[iFrame][(0, 1)].add_new_mol(
                    self.unassignedMolIdx[iFrame][inLeaflet1])
            self.leafletCollection[iFrame][(0, 2)].add_new_mol(
                    self.unassignedMolIdx[iFrame][~inLeaflet1])

        return

