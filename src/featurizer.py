import numpy as np
import pandas as pd
from numba import jit
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from rich.progress import track
from src.io import XVGReader
from src.define import NAME2TYPE

class Featurizer():

    def __init__(self, topFile, trjFile):

        self.topFile = topFile
        self.trjFile = trjFile
        self.u = mda.Universe(topFile, trjFile)
        self.nAtoms : int = len(self.u.atoms)
        self.feature = pd.DataFrame()
        self._get_lipid_type()

    def compute_asphericity_parameter(self, comFile):

        com = XVGReader(comFile).read()
        apCollection = np.array([], dtype=float)
        lCollection = []

        for iFrame in track(range(len(self.u.trajectory))):
            comPos = com.iloc[iFrame].values[1:] * 10
            atomPos = self.u.trajectory[iFrame]._pos
            A = self.__compute_rg_tensor(self.nAtoms, atomPos, comPos)
            l = np.sort(np.linalg.eig(A)[0])[::-1]
            ap = ((l[0] - l[1]) ** 2 + (l[1] - l[2]) ** 2 + (l[2] - l[0]) ** 2) / (2 * (l[0] + l[1] + l[2]) ** 2)
            apCollection = np.append(apCollection, ap)
            lCollection.append(l)
        lCollection = np.array(lCollection)

        self.feature['$A_p$'] = apCollection
        self.feature['$\\lambda_1^2$'] = lCollection[::, 0]
        self.feature['$\\lambda_2^2$'] = lCollection[::, 1]
        self.feature['$\\lambda_3^2$'] = lCollection[::, 2]

        return apCollection, lCollection
    
    def calculate_distance_matrix(self, iFrame, selIdx):

        '''
        Compute the self-pairwise distance of selected atoms
        
        selIdx: the index of the selected atom
        '''

        nAtom = len(selIdx)
        frame = self.u.trajectory[iFrame]
        distance = distances.self_distance_array(frame.positions[selIdx])
        distMatrix = self._transform_to_matrix(distance, nAtom)

        return distMatrix
    
    def _get_lipid_type(self):

        molType = []
        for res in self.u.residues:
            molType.append(NAME2TYPE[res.resname])
        molType = np.array(molType)

        molGroup = {}
        allTypes = set(molType)
        for type_ in allTypes:
            molGroup[type_] = np.argwhere(molType == type_)

        self.molType = molType
        self.molGroup = molGroup
    
    @staticmethod
    @jit(nopython=True)
    def _transform_to_matrix(distance, nAtom):

        distMatrix = np.full((nAtom, nAtom), np.nan)
        iLBound = 0
        nPairs = nAtom - 1 # for atom 0, there are nAtom - 1 pairs
        for iAtom in range(nAtom):
            distMatrix[iAtom, iAtom + 1:] = distance[iLBound:iLBound + nPairs]
            distMatrix[iAtom + 1:, iAtom] = distance[iLBound:iLBound + nPairs]
            iLBound += nPairs
            nPairs -= 1 # for atom i, there are nAtom - i - 1 pairs

        return distMatrix

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _compute_rg_tensor(nAtoms, atomPos, comPos):

        r = atomPos - comPos

        return r.T.dot(r) / nAtoms