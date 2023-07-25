import numpy as np
import pandas as pd
import networkx as nx
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from scipy.sparse.linalg import eigs
from scipy.sparse.lil import lil_matrix
from scipy.sparse.csgraph import laplacian
from rich.progress import track

from src.io import XVGReader
from src.define import NAME2TYPE

class Featurizer():

    def __init__(self, topFile, trjFile, trjInMemorary=False):

        self.topFile = topFile
        self.trjFile = trjFile
        self.u = mda.Universe(topFile, trjFile, in_memory=trjInMemorary)
        self.trjLen = len(self.u.trajectory)
        self.nAtoms : int = len(self.u.atoms)
        self.feature = pd.DataFrame()
        self._get_mol_name_and_type()

    def select_mol(self, selection:str):

        '''
        Select molecule according mol name and mol type
        
        Args:
            selection: name or type of a molecular
        
        Return:
            selMolIdx: a list of the index of selected molecule
        '''

        if selection in self.molName:
            return np.concatenate(np.argwhere(self.idx2name == selection))
        elif selection in self.molType:
            return np.concatenate(np.argwhere(self.idx2type == selection))
        
    def plot_polar_distribution(self, iFrame, selIdx, returnCoord=False):
    
        if len(selIdx.shape) > 1: selIdx = np.concatenate(selIdx)
        pos = np.copy(self.u.trajectory[iFrame].positions)
        com = np.average(pos, axis=0)
        pos -= com
        r, theta, phi = self._cart_to_sph(pos[selIdx, 0], pos[selIdx, 1], pos[selIdx, 2])
        if returnCoord:
            return r, theta, phi
        mirror = self._cube_mirror((np.pi, 2 * np.pi), np.vstack([theta, phi]).T)
        fig, ax = plt.subplots()
        sns.histplot(x=mirror[::, 0], y=mirror[::, 1], ax=ax, weights=1/np.abs(np.sin(mirror[::, 0])), bins=(50, 50))
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\phi$')
        ax.set_ylim((-np.pi, np.pi))
        ax.set_xlim((0, np.pi))

    def _cart_to_sph(self, x, y, z):
  
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)

        return (r, theta, phi)

    def _cube_mirror(self, length, point):

        length = np.array(length)
        point = np.array(point)
        directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)])
        mirrors = [point]
        for direct in directions:
            mirrors.append(point + direct * length)

        return np.concatenate(mirrors)

    def plot_radius_distribution(self, molName: str, iFrame: int, returnDF=False, **kwargs):

        '''
        Plot the radius distribution of a certain kind of molecule
        
        Args:
            molName: The name of molecule that you are interested in.
            iFrame: The index of the frame in the trajectory that you want to plot.
            return_df: Whether to return the dataframe used to plot or not.
            kwargs: kwargs for matplotlib.pyplot.subplots()

        Return:
            fig and ax of matplotlib.pyplot
            If return_df is true, return fig, ax, and df
        '''

        pos = np.copy(self.u.trajectory[iFrame].positions)
        com = np.average(pos, axis=0)
        pos -= com
        # TODO: Add support for select in certain leaflet.
        radius = np.linalg.norm(pos, axis=1)

        atomName = set([atom.name for atom in self.u.select_atoms(f'resname {molName}')])
        selMol = self.u.select_atoms(f'resname {molName}')
        dfPos = pd.DataFrame()
        for atom in atomName:
            selAtom = selMol.select_atoms(f'name {atom}')
            dfPos[atom] = radius[selAtom.ids - 1]
        fig, ax = plt.subplots(**kwargs)
        sns.kdeplot(data=dfPos.sort_index(axis=1), ax=ax)
        ''' 
        ax.set_title(f'{mol} N={len(dfPos)}')
        os.makedirs(f'temp_fig/distribution/Traj-{iTraj}', exist_ok=True)
        plt.savefig(f'temp_fig/distribution/Traj-{iTraj}/{mol}.png', transparent=False)'''

        if returnDF:
            return fig, ax, dfPos
        else:
            return fig, ax
        
    def plot_mass_center_radius_distribution(self, iFrame: int, nMolCutoff: int = 0, noPlot=False):

        pos = np.copy(self.u.trajectory[iFrame].positions)
        com = np.average(pos, axis=0)
        pos -= com
        radius = np.linalg.norm(pos, axis=1)

        if not noPlot:
            fig, ax = plt.subplots()
        legends = []
        df = {}
        for molName in self.molName:
            molIdx = self.select_mol(molName)
            if len(molIdx) < nMolCutoff: continue
            legends.append(molName)
            idxMatrix = []
            for i in molIdx:
                idxMatrix.append(self.u.residues[i].atoms.ids - 1)
            idxMatrix = np.array(idxMatrix)
            selRadius = np.average(radius[idxMatrix], axis=1)
            df[molName] = selRadius
            if not noPlot: sns.kdeplot(data=selRadius)
        if noPlot:
            return df
        plt.legend(legends)
        plt.xlabel(r'Radius (Angstrom)')

        return fig, ax, df

    def compute_asphericity_parameter(self, comFile):

        com = XVGReader(comFile).read()
        apCollection = np.array([], dtype=float)
        lCollection = []

        for iFrame in track(range(self.trjLen)):
            comPos = com.iloc[iFrame].values[1:] * 10
            atomPos = self.u.trajectory[iFrame]._pos
            A = self._compute_rg_tensor(self.nAtoms, atomPos, comPos)
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
    
    def calculate_contact_matrix(self, iFrame, selIdx):

        '''
        Compute the contact matrix of selected atoms
        
        selIdx: the index of the selected atom
        '''

        frame = self.u.trajectory[iFrame]

        return distances.contact_matrix(frame.positions[selIdx],
                                        returntype='sparse')
    
    # TODO: add support for the situtation where contact matrix is not stored

    def calculate_Laplacian_eigenvalue(self, contactMatrix, selIdx, cutoff=1e-8):

        if isinstance(contactMatrix, lil_matrix):
            contactMatrix = contactMatrix.toarray()

        mask = np.full(contactMatrix.shape[0], False)
        mask[selIdx] = True
        l = laplacian(contactMatrix[mask][::, mask].asfptype())
        while True:
            eigVal, _ = eigs(l.asfptype(), k=10, which='SM')
            eigValReal = np.array([val.real for val in eigVal])
            if np.sum(eigValReal > cutoff) > 0:
                break

        return np.sort(eigValReal[eigValReal > cutoff])[0]
    
    def calculate_average_lipid_number(self, contactMatrix, mainIdx, surroundingIdx):

        if isinstance(contactMatrix, lil_matrix):
            contactMatrix = contactMatrix.toarray()

        mainMask = np.full(contactMatrix.shape[0], False)
        mainMask[mainIdx] = True
        surroundingMask = np.full(contactMatrix.shape[0], False)
        surroundingMask[surroundingIdx] = True
        lipidNumber = np.sum(contactMatrix[mainMask][::, surroundingMask], axis=1)

        return np.average(lipidNumber)
    
    def _get_radius_distribution(self, iFrame):

        pos = np.copy(self.u.trajectory[iFrame].positions)
        com = np.average(pos, axis=0)
        pos -= com
        # TODO: Add support for select in certain leaflet.
        radius = np.linalg.norm(pos, axis=1)

        return radius
    
    def _get_mol_name_and_type(self):

        idx2name = []
        idx2type = []
        for res in self.u.residues:
            idx2name.append(res.resname)
            idx2type.append(NAME2TYPE[res.resname])
        idx2name = np.array(idx2name)
        idx2type = np.array(idx2type)

        molCollection = {}
        allName = set(idx2name)
        for name in allName:
            molCollection[name] = np.argwhere(idx2name == name)

        molGroup = {}
        allTypes = set(idx2type)
        for type_ in allTypes:
            molGroup[type_] = np.argwhere(idx2type == type_)

        self.idx2name = np.array(idx2name)
        self.idx2type = np.array(idx2type)
        self.molName = set(self.idx2name)
        self.molType = set(self.idx2type)
        self.molCollection = molCollection
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