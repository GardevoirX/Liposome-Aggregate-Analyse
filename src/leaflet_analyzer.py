import pickle
import warnings
import numpy as np
import mdtraj as md
from rich.progress import track
from src.define import *
from src.io import read_file, write_file
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
        self.notSolventResNum = np.sum(~np.isnan(self.headAtomIdx))
        self.selectedRes = np.arange(self.top.top.n_residues)[
                ~np.isnan(self.headAtomIdx)]
        self.notSelectedRes = np.arange(self.top.top.n_residues)[
                np.isnan(self.headAtomIdx)]
        self.excludedRes = np.array(
                [iRes for iRes, res in enumerate(self.top.top.residues) if res.name in MOLEXCLUDED])
        self.headAtomIdx_noNan = np.array(
                self.headAtomIdx[self.selectedRes], dtype=int)
        self.locationGot = np.full(int(self.args.stop) - int(self.args.start), False)
        self.assignedFrame = np.full(int(self.args.stop) - int(self.args.start), False)
        self.unassignedMolIdx = {}
        self.leafletLocation = {}

    def load_traj(self, stride=None):

        self.traj = md.load(self.trajFileName, top=self.top, stride=stride)

        return self.traj

    def get_leaflet_location(self, leafletType, iFrame=None, start=None, end=None, traj=None):

        if iFrame is not None:
            if not self.locationGot[iFrame]:
                locations = self.__get_leaflet_location(
                        leafletType, [self.leafletCollection[iFrame]], self.traj[iFrame])
                self.leafletLocation[iFrame] = locations[0]
                self.locationGot[iFrame] = True
        elif (start is not None and end is not None) and traj is not None:
            locations = self.__get_leaflet_location(
                    leafletType, self.leafletCollection[start:end], traj)
            for iFrame in range(start, end)[:len(locations)]:
                if not self.locationGot[iFrame]:
                    self.leafletLocation[iFrame] = locations[iFrame - start]
        else:
            raise ValueError('You should input either a start end location and a trajectory or a frame index.')


    def __get_leaflet_location(self, leafletType, leafletCollection, traj, start=None):

        assert len(leafletCollection) == len(traj)
        tempList = []
        for iFrame in range(len(traj)):
            location = []
            if self.__attribute_need_update(iFrame, start, self.locationGot):
                for idx in leafletCollection[iFrame]:
                    atomIdx = np.array(
                            self.headAtomIdx[leafletCollection[iFrame][idx].molIdx])
                    atomIdx = np.array(atomIdx[~np.isnan(atomIdx)], dtype=int)        
                    if leafletType == 'membrane':
                        location.append(np.average(traj.xyz[iFrame][atomIdx][::, -1]))
            else:
                location.append(None)
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

        #assert len(assignedMolIdx) + len(self.unassignedMolIdx[iFrame]) == self.totalResNum

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
                for iFrame in range(start, end)[:len(traj)]:
                    tempUnassignedMolIdx.append(self.unassignedMolIdx[iFrame])
                    tempLeafletLocation.append(self.leafletLocation[iFrame])
                results = self.__assign_molecules(
                        leafletType, tempLeafletLocation, tempUnassignedMolIdx, traj, start)
                for iFrame in range(start, end)[:len(traj)]:
                    if not self.assignedFrame[iFrame]:
                        
                        self.leafletCollection[iFrame][(0, 1)].add_new_mol(
                                self.unassignedMolIdx[iFrame][results[iFrame - start]])
                        self.leafletCollection[iFrame][(0, 2)].add_new_mol(
                                self.unassignedMolIdx[iFrame][~results[iFrame - start]])
                        self.assignedFrame[iFrame] = True
            else:
                raise ValueError('You should input either a start end location and a trajectory or a frame index.')

    def __assign_molecules(self, leafletType, leafletLocation, unassignedMolIdx, traj, start):

        # TODO: assign molecules according to its z coordinate or the distance to vesicle center
        assert (len(unassignedMolIdx) == len(traj)) and (len(leafletLocation) == len(traj))
        tempList = []
        if leafletType == 'membrane':
            for iFrame in range(len(traj)):
                headAtomIdx = []
                if self.assignedFrame[iFrame + start]:
                    tempList.append([None])
                    continue
                for iRes in unassignedMolIdx[iFrame]:
                    headAtomIdx.append(self.top.top.residue(iRes).atom(0).index)
                    #select the head bead of the molecule
                headAtomCoord = traj.xyz[iFrame][headAtomIdx][::, -1]
                inLeaflet1 = abs(headAtomCoord - leafletLocation[iFrame][0])\
                        < abs(headAtomCoord - leafletLocation[iFrame][1])
                tempList.append(inLeaflet1)

        return tempList

    def merge_two_leaflet(self, iFrame, iAgg, location1, location2):

        assert location1 != location2
        
        leaflet1 = self.leafletCollection[iFrame][(iAgg, location1)]
        leaflet2 = self.leafletCollection[iFrame][(iAgg, location2)]
        
        if leaflet1.nMol < leaflet2.nMol:
            leaflet1, leaflet2 = (leaflet2, leaflet1)
        leaflet1.add_new_mol(leaflet2.molIdx)
        
        return iFrame, (leaflet2.iAgg, leaflet2.location)

    def remove_waste_leaflet(self, iFrame, idx):
        
        del self.leafletCollection[iFrame][idx]
        
        newAgg = {}
        for iAgg, location in self.leafletCollection[iFrame]:
            if iAgg != idx[0]:
                newAgg[(iAgg, location)] = self.leafletCollection[iFrame][(iAgg, location)]
            elif iAgg == idx[0]:
                if location > idx[1]:
                    newAgg[(iAgg, location - 1)] = self.leafletCollection[iFrame][(iAgg, location)]
                    newAgg[(iAgg, location - 1)].location = location - 1
                else:
                    newAgg[(iAgg, location)] = self.leafletCollection[iFrame][(iAgg, location)]
        self.leafletCollection[iFrame] = newAgg
        self.locationGot[iFrame] = False
        
        return

    def reindex(self, leafletType):
    
        for iFrame in track(range(len(self.leafletCollection) - 1)):
            if leafletType == 'membrane':
                currentLeaflet1 = set(self.leafletCollection[iFrame][(0, 1)].molIdx)
                currentLeaflet2 = set(self.leafletCollection[iFrame][(0, 2)].molIdx)
                nextLeaflet1 = set(self.leafletCollection[iFrame+1][(0, 1)].molIdx)
                if len(nextLeaflet1.intersection(currentLeaflet1)) < len(nextLeaflet1.intersection(currentLeaflet2)):
                    self.leafletCollection[iFrame+1][(0, 1)], self.leafletCollection[iFrame+1][(0, 2)] = \
                            self.leafletCollection[iFrame+1][(0, 2)], self.leafletCollection[iFrame+1][(0, 1)]
        
        return

    def __attribute_need_update(self, iFrame, start, boolAry):

        if start is None:
            return boolAry[iFrame]
        else:
            return boolAry[iFrame + start]