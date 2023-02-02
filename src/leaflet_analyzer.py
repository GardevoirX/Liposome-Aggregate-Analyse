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
        self.totalResNum = len(self.headAtomIdx)
        self.selectedRes = np.arange(self.top.top.n_residues)[
                ~np.isnan(self.headAtomIdx)]
        self.notSelectedRes = np.arange(self.top.top.n_residues)[
                np.isnan(self.headAtomIdx)]
        self.headAtomIdx_noNan = np.array(
                self.headAtomIdx[self.selectedRes], dtype=int)
        self.assignedFrame = np.full(int(self.args.stop) - int(self.args.start), False)
        self.unassignedMolIdx = {}
        self.leafletLocation = {}

    def load_traj(self, stride=None):

        self.traj = md.load(self.trajFileName, top=self.top, stride=stride)

        return self.traj

    def get_leaflet_location(self, leafletType, iFrame=None, start=None, end=None, traj=None):

        if iFrame is not None:
            locations = self.__get_leaflet_location(
                    leafletType, [self.leafletCollection[iFrame]], self.traj[iFrame])
            self.leafletLocation[iFrame] = locations[0]
        elif (start is not None and end is not None) and traj is not None:
            locations = self.__get_leaflet_location(
                    leafletType, self.leafletCollection[start:end], traj)
            for iFrame in range(start, end):
                self.leafletLocation[iFrame] = locations[iFrame - start]
        else:
            raise ValueError('You should input either a start end location and a trajectory or a frame index.')


    def __get_leaflet_location(self, leafletType, leafletCollection, traj):

        assert len(leafletCollection) == len(traj)
        tempList = []
        for iFrame in range(len(traj)):
            location = []
            for idx in leafletCollection[iFrame]:
                atomIdx = np.array(
                        self.headAtomIdx[leafletCollection[iFrame][idx].molIdx])
                atomIdx = np.array(atomIdx[~np.isnan(atomIdx)], dtype=int)        
                if leafletType == 'membrane':
                    location.append(np.average(traj.xyz[iFrame][atomIdx][::, -1]))
            tempList.append(location)

        return tempList

    def get_aggregate_molecules(self):

        self.aggAtomIdx = self.top.top.select('not resname ' + ' '.join(RESEXCLUDED))
        self.aggResIdx = [iRes for iRes, res in enumerate(self.top.top.residues) if not res.name in MOLEXCLUDED]

    def find_unassigned_molecules(self, iFrame):

        resIdx = np.full(self.totalResNum, True)
        assignedMolIdx = np.concatenate(
            [self.leafletCollection[iFrame][iter].molIdx 
                for iter in self.leafletCollection[iFrame]])
        resIdx[assignedMolIdx] = False
        self.unassignedMolIdx[iFrame] = []
        for idx, notSelected in enumerate(resIdx):
            if notSelected and (self.top.top.residue(idx).name not in MOLEXCLUDED):
                self.unassignedMolIdx[iFrame].append(idx)
        self.unassignedMolIdx[iFrame] = np.array(
                self.unassignedMolIdx[iFrame])

        assert len(assignedMolIdx) + len(self.unassignedMolIdx[iFrame]) == self.totalResNum

        return self.unassignedMolIdx[iFrame]

    def assign_molecules(self, leafletType, iFrame=None, start=None, end=None, traj=None):

        if leafletType == 'membrane':
            if iFrame is not None:
                if not self.assignedFrame[iFrame]:
                    results = self.__assign_molecules(
                            leafletType, [self.leafletLocation[iFrame]], 
                            [self.unassignedMolIdx[iFrame]], self.traj[iFrame])
                    self.leafletCollection[iFrame][(0, 1)].add_new_mol(
                            self.unassignedMolIdx[iFrame][results[0]])
                    self.leafletCollection[iFrame][(0, 2)].add_new_mol(
                            self.unassignedMolIdx[iFrame][~results[0]])
                    self.assignedFrame[iFrame] = True
            elif (start is not None and end is not None) and traj is not None:
                tempUnassignedMolIdx = []
                tempLeafletLocation = []
                for iFrame in range(start, end):
                    tempUnassignedMolIdx.append(self.unassignedMolIdx[iFrame])
                    tempLeafletLocation.append(self.leafletLocation[iFrame])
                results = self.__assign_molecules(
                        leafletType, tempLeafletLocation, tempUnassignedMolIdx, traj)
                for iFrame in range(start, end):
                    if not self.assignedFrame[iFrame - start]:
                        self.leafletCollection[iFrame][(0, 1)].add_new_mol(
                                self.unassignedMolIdx[iFrame][results[iFrame - start]])
                        self.leafletCollection[iFrame][(0, 2)].add_new_mol(
                                self.unassignedMolIdx[iFrame][~results[iFrame - start]])
                        self.assignedFrame[iFrame - start] = True
            else:
                raise ValueError('You should input either a start end location and a trajectory or a frame index.')

    def __assign_molecules(self, leafletType, leafletLocation, unassignedMolIdx, traj):

        # TODO: assign molecules according to its z coordinate or the distance to vesicle center
        assert (len(unassignedMolIdx) == len(traj)) and (len(leafletLocation) == len(traj))
        tempList = []
        if leafletType == 'membrane':
            for iFrame in range(len(traj)):
                headAtomIdx = []
                for iRes in unassignedMolIdx[iFrame]:
                    headAtomIdx.append(self.top.top.residue(iRes).atom(0).index)
                    #select the head bead of the molecule
                headAtomCoord = traj.xyz[iFrame][headAtomIdx][::, -1]
                inLeaflet1 = abs(headAtomCoord - leafletLocation[iFrame][0])\
                        < abs(headAtomCoord - leafletLocation[iFrame][1])
                tempList.append(inLeaflet1)

        return tempList

