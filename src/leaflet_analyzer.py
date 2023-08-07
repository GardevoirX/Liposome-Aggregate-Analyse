from typing import Union
import numpy as np
import pandas as pd
import MDAnalysis as mda
from sklearn import neighbors
from rich.progress import track
from src.define import *
from src.io import read_file, write_file
from src.selection import *
from src.utils.dist import cal_distance_matrix, select_center_based_on_kcenter

class Analyzer():

    def __init__(self, fileName, argsFileName):

        self.leafletCollection = read_file(fileName)
        self.args = read_file(argsFileName)
        self.trajFileName = self.args.trajFile
        self.top = mda.Universe(self.args.topFile)
        self.traj = mda.Universe(self.args.topFile, self.args.trajFile).trajectory
        self.molType = self.args.molType
        self.headAtomSelection = fix_not_selected_mol(self.top, \
                get_atom_selection(self.args.selHeadAtom))
        self.headAtomIdx = get_index_of_selected_atom(self.top, \
                self.headAtomSelection)
        self.tailAtomSelection = fix_not_selected_mol(self.top, \
                get_atom_selection(self.args.selTailAtom))
        self.tailAtomIdx = get_index_of_selected_atom(self.top, \
                self.tailAtomSelection)
        self.totalResNum = len(self.headAtomIdx)
        self.notSolventResNum = np.sum(~np.isnan(self.headAtomIdx))
        self.selectedRes = np.arange(self.totalResNum)[
                ~np.isnan(self.headAtomIdx)]
        self.notSelectedRes = np.arange(self.totalResNum)[
                np.isnan(self.headAtomIdx)]
        self.excludedRes = np.array(
                [iRes for iRes, res in enumerate(self.top.residues) if res.resname in MOLEXCLUDED])
        self.headAtomIdx_noNan = np.array(
                self.headAtomIdx[self.selectedRes], dtype=int)
        self.headAtomIdx[self.notSelectedRes] = assign_head_atom_for_not_selected_mol(
                self.top, self.notSelectedRes)
        self.headAtomIdx = np.array(self.headAtomIdx, dtype=int)
        self.tailAtomIdx_noNan = np.array(
                self.tailAtomIdx[self.selectedRes], dtype=int)
        self.locationGot = np.full(int(self.args.stop) - int(self.args.start), False)
        self.assignedFrame = np.full(int(self.args.stop) - int(self.args.start), False)
        self.unassignedMolIdx = {}

    def get_leaflet_location(self, leafletType, iFrame=None, start=None, end=None, traj=None):

        if iFrame is not None:
            if not self.locationGot[iFrame]:
                self._get_leaflet_location(
                        leafletType, [self.leafletCollection[iFrame]], self.traj[iFrame:iFrame+1])
                self.locationGot[iFrame] = True
        elif (start is not None and end is not None) and traj is not None:
            self._get_leaflet_location(
                    leafletType, self.leafletCollection[start:end], traj)
        else:
            raise ValueError('You should input either a start end location and a trajectory or a frame index.')


    def _get_leaflet_location(self, leafletType, leafletCollection, traj, start=None):

        tempList = []
        for iFrame in range(len(traj)):
            if self.__attribute_need_update(iFrame, start, self.locationGot):
                for idx in leafletCollection[iFrame]:
                    atomIdx = np.array(
                            self.headAtomIdx[leafletCollection[iFrame][idx].molIdx])
                    # remove np.nan
                    atomIdx = np.array(atomIdx[~np.isnan(atomIdx)], dtype=int)
                    leafletCollection[iFrame][idx].leafletType = leafletType
                    if leafletType == 'membrane':
                        leafletCollection[iFrame][idx].location = np.average(traj[iFrame].positions[atomIdx][::, -1])
                    elif leafletType == 'vesicle':
                        center = np.average(traj[iFrame].positions[atomIdx], axis=0)
                        leafletCollection[iFrame][idx].center = center
                        radius = np.linalg.norm(traj[iFrame].positions[atomIdx] - center, axis=1)
                        leafletCollection[iFrame][idx].location = np.average(radius)
            else:
                continue

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
            if notSelected and (self.top.residues[idx].resname not in MOLEXCLUDED):
                self.unassignedMolIdx[iFrame].append(idx)
        self.unassignedMolIdx[iFrame] = np.array(
                self.unassignedMolIdx[iFrame])

        #assert len(assignedMolIdx) + len(self.unassignedMolIdx[iFrame]) == self.totalResNum

        return self.unassignedMolIdx[iFrame]

    def assign_molecules(self, iFrame):

        if not self.assignedFrame[iFrame]:
            self._assign_molecules( 
                    self.unassignedMolIdx[iFrame], iFrame)

    def _assign_molecules(self, unassignedMolIdx, iFrame):

        # TODO: assign molecules according to its z coordinate or the distance to vesicle center
        actualIdx = iFrame
        if hasattr(self.leafletCollection[actualIdx][(0, 1)], 'assigned'):
            return
        if len(self.leafletCollection[actualIdx].keys()) == 2 and \
            hasattr(self.leafletCollection[actualIdx][(0, 1)], 'leafletType'):
            self._assign_based_on_leaflet_location(actualIdx, unassignedMolIdx)
        else:
            self._assign_based_on_neighbour(actualIdx, unassignedMolIdx)

    def _assign_based_on_leaflet_location(self, actualIdx, unassignedMolIdx):

        assert hasattr(self.leafletCollection[actualIdx][(0, 1)], 'location') and \
                hasattr(self.leafletCollection[actualIdx][(0, 2)], 'location'), \
                'Please call get_leaflet_location() and specify the leaflet type first!'
        location0 = self.leafletCollection[actualIdx][(0, 1)].location
        location1 = self.leafletCollection[actualIdx][(0, 2)].location
        leafletType = self.leafletCollection[actualIdx][(0, 1)].leafletType
        headAtomIdx = []
        # select the head bead of the molecule
        for iRes in unassignedMolIdx:
            headAtomIdx.append(self.top.residues[iRes].atoms[0].index)
        if leafletType == 'membrane':
            headAtomCoord = self.traj[actualIdx] \
                                .positions[headAtomIdx][::, -1]
        elif leafletType == 'vesicle':
            headAtomCoord = \
                    np.linalg.norm(self.traj[actualIdx].positions[headAtomIdx] -\
                    self.leafletCollection[actualIdx][(0, 1)].center, axis=1)
        inLeaflet1 = abs(headAtomCoord - location0)\
                < abs(headAtomCoord - location1)
        self.leafletCollection[actualIdx][(0, 1)] \
            .add_new_mol(unassignedMolIdx[inLeaflet1])
        self.leafletCollection[actualIdx][(0, 2)] \
            .add_new_mol(unassignedMolIdx[~inLeaflet1])
        self.leafletCollection[actualIdx][(0, 1)].get_composition(self.molType)
        self.leafletCollection[actualIdx][(0, 2)].get_composition(self.molType)
        self.leafletCollection[actualIdx][(0, 1)].assigned = True
        self.leafletCollection[actualIdx][(0, 2)].assigned = True

    def _assign_based_on_neighbour(self, actualIdx, unassignedMolIdx):

        assigned = np.concatenate([self.leafletCollection[actualIdx][key].molIdx \
                                  for key in self.leafletCollection[actualIdx].keys()])
        X = self.traj[actualIdx]._pos[np.array(self.headAtomIdx[assigned], dtype=int)]
        y = np.concatenate([np.full(len(self.leafletCollection[actualIdx][key].molIdx), i) \
                            for i, key in enumerate(self.leafletCollection[actualIdx].keys())])
        clf = neighbors.KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        result = clf.predict(self.traj[actualIdx]._pos[
                np.array(self.headAtomIdx[unassignedMolIdx], dtype=int)])
        # update
        for i, key in enumerate(self.leafletCollection[actualIdx].keys()):
            self.leafletCollection[actualIdx][key].add_new_mol(unassignedMolIdx[result == i])
            self.leafletCollection[actualIdx][key].get_composition(self.molType)
            self.leafletCollection[actualIdx][key].assigned = True


    def calculate_flip_flop_times(self, selIdx:np.array, startFrame:int):
        """
        Calculate the flip flop times based on the given selection index and start frame.

        Parameters:
            selIdx (np.array): The selection index.
            startFrame (int): The starting frame.

        Returns:
            int: The total number of flip flop times.
        """

        status = np.full((20001, len(selIdx)), np.nan)
        for iFrame in track(range(startFrame, 20001)):
            center = np.average([self.leafletCollection[iFrame][(0, 1)].center, self.leafletCollection[iFrame][(0, 2)].center], axis=0)
            radius = np.linalg.norm(self.traj[iFrame]._pos[selIdx] - center, axis=1)
            location1 = self.leafletCollection[iFrame][(0, 1)].location
            location2 = self.leafletCollection[iFrame][(0, 2)].location
            thickness = abs(location1 - location2)
            bufferZone = (min(location1, location2) + 0.1 * thickness,
                        max(location1, location2) - 0.1 * thickness)
            status[iFrame, radius < bufferZone[0]] = 1
            status[iFrame, radius > bufferZone[1]] = 2

        status = pd.DataFrame(status).fillna(method='ffill').fillna(method='bfill')

        return ((status - status.shift(1)).fillna(0) != 0).sum().sum()
    
    def calculate_correlation(self, startFrame: int, endFrame: int, nClusters: int, atomIdx: list, resIdx: list):

        corr = []
        df = pd.DataFrame()
        df['atom_ids'] = atomIdx
        df['res_ids'] = resIdx
        for iFrame in track(range(startFrame, endFrame)):
            clusterCenters = select_center_based_on_kcenter(nClusters, 
                    self.traj[iFrame]._pos[self.tailAtomIdx_noNan])
            '''r = (self.leafletCollection[iFrame][(0, 1)].location + self.leafletCollection[iFrame][(0, 2)].location) / 2
            clusterPos = clusterCenters * r'''
            positions = np.array(self.traj[iFrame]._pos[df['atom_ids']], dtype=float)
            df['labels'] = self._assign_molecule_to_cluster(positions, np.array(clusterCenters, dtype=float))
            corr.append(self._cal_corrcoef(iFrame, df))

        return corr

    def merge_two_leaflet(self, iFrame, iAgg, location1, location2):
        """
        Merge two leaflets based on their locations.

        Args:
            iFrame (int): The index of the frame.
            iAgg (int): The index of the aggregate.
            location1 (str): The location of the first leaflet.
            location2 (str): The location of the second leaflet.

        Returns:
            tuple: A tuple containing the updated iFrame and the merged leaflet information.
        """
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
    
    def _assign_molecule_to_cluster(self, positions: np.ndarray, selPoints: np.ndarray):

        center = np.mean(positions, axis=0)
        positions -= center
        distMat = cal_distance_matrix(positions, selPoints)
            
        return np.argmin(distMat, axis=1)
    
    def _cal_corrcoef(self, iFrame: int, df: pd.DataFrame):
    
        lipidLocation = np.zeros(len(self.top.residues), dtype=int)
        lipidLocation[self.leafletCollection[iFrame][(0, 1)].molIdx] = 1
        lipidLocation[self.leafletCollection[iFrame][(0, 2)].molIdx] = 2
        df['location'] = lipidLocation[df['res_ids']]
        dfGroupby = df.groupby(['location', 'labels']).count().unstack(fill_value=0).stack()

        count1 = dfGroupby.loc[1, :]['res_ids']
        count2 = dfGroupby.loc[2, :]['res_ids']
        
        return np.corrcoef(count1, count2)[0, 1]

    def __attribute_need_update(self, iFrame, start, boolAry):

        if start is None:
            return not boolAry[iFrame]
        else:
            return not boolAry[iFrame + start]