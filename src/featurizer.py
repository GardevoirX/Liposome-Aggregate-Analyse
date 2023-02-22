import numpy as np
import pandas as pd
from numba import jit
import MDAnalysis as mda
from rich.progress import track
import sys
sys.path.append('/share/home/qjxu/scripts/aggregate_analyse')
from src.io import XVGReader

class Featurizer():

    def __init__(self, topFile, trjFile):

        self.topFile = topFile
        self.trjFile = trjFile
        self.u = mda.Universe(topFile, trjFile)
        self.nAtoms : int = len(self.u.atoms)
        self.feature = pd.DataFrame()

        pass

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

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __compute_rg_tensor(nAtoms, atomPos, comPos):

        r = atomPos - comPos

        return r.T.dot(r) / nAtoms