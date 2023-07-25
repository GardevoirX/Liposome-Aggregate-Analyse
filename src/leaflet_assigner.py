import os
import numpy as np
import cupy as cp
import open3d as o3d
import networkx as nx
import MDAnalysis as mda
from MDAnalysis.analysis.distances import self_distance_array
from numba import jit
from itertools import combinations
from math import ceil
from src.leaflet import Leaflet
from src.selection import *
from src.io import write_file


class LeafletAssigner():

    '''Assign molecules in a aggregation into several leaflets

    Args:
    topFile: The topology file of the trajectory you want to analyze
    trajFile: The trajectory file
    selHeadAtom:
    selTailAtom:
    '''

    def __init__(self, topFile, trajFile, \
                 selHeadAtom, selTailAtom, allSetsFile=None, \
                 start=None, stop=None, skip=None, trjSkip=1,\
                 distanceCutoff=16.6, parallelDegreeCutoff=14, \
                 minLeafletSize=30, chunkSize=100, outputPref='./'):

        self.topFile = topFile
        self.trajFile = trajFile
        self.top = mda.Universe(topFile, trajFile)
        self.nRes = len(self.top.residues)
        self.traj = self.top.trajectory
        self.trajLen = len(self.traj)
        self.selHeadAtomCmd = selHeadAtom
        self.selTailAtomCmd = selTailAtom
        self.selHeadAtom = get_atom_selection(selHeadAtom)
        self.selTailAtom = get_atom_selection(selTailAtom)
        self._prepare_info(self.top, self.selHeadAtom, self.selTailAtom)
        self.start = start if start is not None else 0
        self.stop = stop if stop is not None else self.trajLen
        self.skip = skip if skip is not None else 1
        self.allSetsFile = allSetsFile
        self.allSets = self._read_allSets(allSetsFile, self.start, self.stop, self.skip, trjSkip) \
                if isinstance(self.allSetsFile, str) \
                else self._generate_allSets(self.selRes, self.start, self.stop, self.skip, trjSkip)
        self.nFrames = len(self.allSets)
        self.trjSkip = trjSkip
        self.distanceCutoff = distanceCutoff
        self.parallelDegreeCutoff = parallelDegreeCutoff
        self.minLeafletSize = minLeafletSize
        self.chunkSize = chunkSize
        self.nChunk = ceil(self.nFrames / chunkSize)
        self.outputPref = outputPref
        os.makedirs(self.outputPref, exist_ok=True)

    def run(self):

        self.trajToAnalysis = self.traj[self.start:self.stop:self.skip]
        self.leafIdx = []
        for iChunk in range(self.nChunk):
            start = iChunk * self.chunkSize
            stop = (iChunk + 1) * self.chunkSize if iChunk < self.nChunk - 1 \
                    else self.nFrames
            aggGraph = self.generate_adjacency_matrix_of_atom_pairs(start, stop)
            normalVector = self.get_normal_vector_of_every_molecular(start, stop)
            assignment = self.assign_leaflet(aggGraph, normalVector)
            self.leafIdx.append(self.assign_leaflet_index(
                    assignment, start, stop))

        self.result = self.leafIdx = np.concatenate(self.leafIdx)
        self._write_results()

        return self.leafIdx           

    def generate_adjacency_matrix_of_atom_pairs(self, start, stop):
        
        nSelectedMol = len(self.headAtomIdx)
        aggGraph = []
        #dist = []
        for iFrame in range(start, stop):
            # Assuming that a molecular cannot transfer from one aggregate to another unless two aggregates fuse
            if (iFrame == start) or \
               (len(self.allSets[iFrame]) \
                != len(self.allSets[iFrame - 1])):
                molPairs, atomPairs = self._update_pair_info(iFrame)

            # TODO: optimize for many aggregates situtation
            pairDistances = [self_distance_array(
                self.trajToAnalysis[iFrame].positions[self.headAtomIdx[agg]])
                for agg in self.allSets[iFrame]]
            #dist.append(np.concatenate(pairDistances))
            selectedPairs = np.concatenate(pairDistances) \
                < self.distanceCutoff
            aggGraph.append(
                self._generate_contact_graph(
                    molPairs[selectedPairs], nSelectedMol))

        return aggGraph
    
    def get_normal_vector_of_every_molecular(self, start, stop):

        normalVector = np.zeros((stop - start, self.nRes, 3))
        subTraj = np.array([np.copy(self.trajToAnalysis[iFrame].positions)
                            for iFrame in range(start, stop)])
        molecularOrientation, headPos = self._get_mol_orientation_CUDA(subTraj, 
                self.head_resIdxMatrix, self.tail_resIdxMatrix, 
                self.nHeadAtomPerMol, self.nTailAtomPerMol)
        for iFrame in range(start, stop):
            #print('Computing the normal vector of Frame {}'.format(iFrame))
            normals = self._calculate_normal_vector_from_cloud_point(headPos[iFrame - start][self.selRes])
            toConvert = np.argwhere((self._product(molecularOrientation[iFrame - start][self.selRes], normals)) < 0)
            normals[toConvert] = -normals[toConvert] 
            normalVector[iFrame - start][self.selRes] = normals

        return normalVector
    
    def _calculate_normal_vector_from_cloud_point(self, position):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(position)
        pcd.estimate_normals(search_param=
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.distanceCutoff, max_nn=40))
        pcd.normalize_normals()

        return np.asarray(pcd.normals)
    
    def assign_leaflet(self, aggGraph, normalVector):
        
        leafletAssignment = []
        parallelDegree = np.cos(np.deg2rad(self.parallelDegreeCutoff))
        for iFrame in range(len(aggGraph)):
            pairs = np.array(aggGraph[iFrame].edges())
            productOfVector1 = self._product(normalVector[iFrame][pairs[::, 0]], normalVector[iFrame][pairs[::, 1]])
            #productOfVector2 = self._product(normalVector[iFrame][pairs[::, 0]], -normalVector[iFrame][pairs[::, 1]])
            pairsNotParallel = (productOfVector1 < parallelDegree)# & (productOfVector2 < parallelDegree)
            aggGraph[iFrame].remove_edges_from(pairs[pairsNotParallel])
            assign = []
            for g in nx.connected_components(aggGraph[iFrame]):
                if len(g) > self.minLeafletSize: 
                    assign.append(np.array(list(g)))
            leafletAssignment.append(assign)

        return leafletAssignment
    
    def assign_leaflet_index(self, assignment, start, stop):

        leafletIdx = []
        for iFrame in range(start, stop):
            leafletIdx.append({})
            leafletsInAgg = np.full(len(self.allSets[iFrame]), 1)
            for leaflet in assignment[iFrame - start]:
                for iAgg in range(len(self.allSets[iFrame])):
                    if leaflet[0] in self.allSets[iFrame][iAgg]:
                            leafletIdx[iFrame - start][(iAgg, leafletsInAgg[iAgg])] = Leaflet(iFrame, iAgg, leafletsInAgg[iAgg], leaflet)
                            leafletIdx[iFrame - start][(iAgg, leafletsInAgg[iAgg])].get_composition(self.molType)
                            leafletsInAgg[iAgg] += 1

        return leafletIdx


    def _prepare_info(self, top, selHeadAtom, selTailAtom):

        self.molType = self._get_mol_type(top)
        selHeadAtom = fix_not_selected_mol(top, selHeadAtom)
        selTailAtom = fix_not_selected_mol(top, selTailAtom)
        self.headAtomIdx = np.array(get_index_of_selected_atom(top, selHeadAtom),
                                    dtype=int)
        head_resIdxMatrix, nHeadAtomPerMol \
                = self._get_atom_residue_matrix(top, selHeadAtom, self.molType)
        tail_resIdxMatrix, nTailAtomPerMol \
                = self._get_atom_residue_matrix(top, selTailAtom, self.molType)
        self.selRes = get_selected_residue(top, selHeadAtom)
        self.nSelRes = len(self.selRes)
        self.head_resIdxMatrix, self.nHeadAtomPerMol, \
        self.tail_resIdxMatrix, self.nTailAtomPerMol \
                = cp.array(head_resIdxMatrix), cp.array(nHeadAtomPerMol), \
                cp.array(tail_resIdxMatrix), cp.array(nTailAtomPerMol)

    @staticmethod
    def _get_mol_type(top):
    
        return np.array([mol.resname for mol in top.residues if not mol.resname in RESEXCLUDED])
    
    @staticmethod
    def _get_atom_residue_matrix(traj, selAtom, molType):
    
        ref = np.array(range(len(traj.atoms)))
        atom_resIdxMatrix = np.array(
            [np.isin(ref, 
                np.array([atom.index 
                          for atom in res.atoms if atom.name in selAtom[res.resname]]))
             for res in traj.residues if not res.resname in RESEXCLUDED])
        nAtomPerMol = []
        for molName in molType:
            if len(selAtom[molName]):
                nAtomPerMol.append(1 / len(selAtom[molName]))
            else:
                nAtomPerMol.append(np.nan)
        nAtomPerMol = np.array([nAtomPerMol]).T
        
        return atom_resIdxMatrix, nAtomPerMol

    def _read_allSets(self, fileName, start, stop, skip, trjSkip):
    
        '''
            In case the traj you provided is processed with `gmx trjconv -skip`
            while the all_sets file is generated based on the original whole traj,
            you are required to provide trjSkip.

            The format of all_sets is:
            ................................
            F 1
            1 2 3 4 5 6 7 8 9
            12 13 15 16
            10 11 17 18 19 20
            F 2
            ................................
        '''
        start = start * trjSkip
        stop = stop * trjSkip

        with open(fileName, 'r') as f:
            allSets = {}
            iFrame = -1
            for line in f.readlines():
                if line[0] == 'F':
                    iFrame += 1
                    allSets[iFrame] = []
                else:
                    allSets[iFrame].append(list(np.array(line.split(), dtype='int')))
        allSets = list([allSets[key] for key in allSets.keys()])[start:stop:skip * trjSkip]

        return allSets

    @staticmethod
    def _generate_allSets(selRes, start, stop, skip, trjSkip):

        # TODO: adapt for trjSkip != 1
        selRes = np.array(selRes)
        allSets = np.array([[selRes] for _ in range(start, stop)])[::skip]

        return allSets
    
    def _cartesian_product(self, array):

        if len(array) == 1:
            return [(array[0], array[0])]
        else:
            return list(combinations(array, 2))
        
    def _update_pair_info(self, iFrame):

        molPairs = np.concatenate([
            self._cartesian_product(agg) for agg in self.allSets[iFrame]])
        atomPairs = self.headAtomIdx[molPairs]

        return molPairs, atomPairs
    
    def _generate_contact_graph(self, molPairs, nSelectedMol):

        G = nx.Graph()
        G.add_nodes_from(range(nSelectedMol))
        nx.from_edgelist(molPairs, G)

        return G
    
    @staticmethod
    def _get_mol_orientation_CUDA(xyz, head_resIdxMatrix, tail_resIdxMatrix, nHeadAtomPerMol, nTailAtomPerMol):

        #print('Now getting the molecular orientation.')
        xyz = cp.array(xyz)
        headPos = cp.dot(head_resIdxMatrix, xyz).transpose((1, 0, 2))
        headPos = cp.multiply(headPos, nHeadAtomPerMol)
        tailPos = cp.dot(tail_resIdxMatrix, xyz).transpose((1, 0, 2))
        tailPos = cp.multiply(tailPos, nTailAtomPerMol)
        molecularOrientation = cp.subtract(tailPos, headPos)
        molecularOrientation /= cp.expand_dims(cp.linalg.norm(molecularOrientation, axis=2), 2)

        return cp.asnumpy(molecularOrientation), cp.asnumpy(headPos)
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _product(v1, v2):
        
        return np.sum(v1 * v2, axis=1)
    
    def _write_results(self):

        write_file(self.leafIdx, os.path.join(self.outputPref, 'leaflet.pickle'))
        args = self.Arg()
        args.trajFile = self.trajFile
        args.topFile = self.topFile
        args.selHeadAtom = self.selHeadAtomCmd
        args.stop = self.stop
        args.start = self.start
        args.molType = self.molType
        write_file(args, os.path.join(self.outputPref, 'leaflet_args.pickle'))


    class Arg():

        pass


if __name__ == '__main__':
    assigner = LeafletAssigner(topFile='example/dry.gro',
                               trajFile='example/short_traj.xtc',
                               selHeadAtom='APC:GL2|BNSM:AM1 AM2|DAPC:GL1 GL2|DAPE:GL1 GL2|DAPS:GL1 GL2|DBSM:AM1 AM2|DOPC:GL1 GL2|DOPE:GL1 GL2|DPCE:AM1 AM2|DPG1:AM1 AM2|DPG3:AM1 AM2|DPSM:AM1 AM2|DUPE:GL1 GL2|DUPS:GL1 GL2|DXCE:AM1 AM2|DXG1:AM1 AM2|DXG3:AM1 AM2|DXSM:AM1 AM2|IPC:GL2|OPC:GL2|PADG:GL1 GL2|PAPA:GL1 GL2|PAPC:GL1 GL2|PAPE:GL1 GL2|PAPI:GL1 GL2|PAPS:GL1 GL2|PEPC:GL1 GL2|PGSM:AM1 AM2|PIDG:GL1 GL2|PIPA:GL1 GL2|PIPC:GL1 GL2|PIPE:GL1 GL2|PIPI:GL1 GL2|PIPS:GL1 GL2|PIPX:GL1 GL2|PNCE:AM1 AM2|PNG1:AM1 AM2|PNG3:AM1 AM2|PNSM:AM1 AM2|PODG:GL1 GL2|POP1:GL1 GL2|POP2:GL1 GL2|POP3:GL1 GL2|POPA:GL1 GL2|POPC:GL1 GL2|POPE:GL1 GL2|POPI:GL1 GL2|POPS:GL1 GL2|POPX:GL1 GL2|POSM:AM1 AM2|PPC:GL2|PQPE:GL1 GL2|PQPS:GL1 GL2|PUDG:GL1 GL2|PUPA:GL1 GL2|PUPC:GL1 GL2|PUPE:GL1 GL2|PUPI:GL1 GL2|PUPS:GL1 GL2|UPC:GL2|XNCE:AM1 AM2|XNG1:AM1 AM2|XNG3:AM1 AM2|XNSM:AM1 AM2',
                               selTailAtom='APC:C5B|BNSM:C4A C6B|DAPC:C5A C5B|DAPE:C5A C5B|DAPS:C5A C5B|DBSM:C4A C5B|DOPC:C4A C4B|DOPE:C4A C4B|DPCE:C3A C4B|DPG1:C3A C4B|DPG3:C3A C4B|DPSM:C3A C4B|DUPE:D5A D5B|DUPS:D5A D5B|DXCE:C5A C6B|DXG1:C5A C6B|DXG3:C5A C6B|DXSM:C5A C6B|IPC:C4B|OPC:C4B|PADG:C5A C4B|PAPA:C5A C4B|PAPC:C5A C4B|PAPE:C5A C4B|PAPI:C5A C4B|PAPS:C5A C4B|PEPC:C5A C4B|PGSM:C3A C5B|PIDG:C4A C4B|PIPA:C4A C4B|PIPC:C4A C4B|PIPE:C4A C4B|PIPI:C4A C4B|PIPS:C4A C4B|PIPX:C4A C4B|PNCE:C3A C6B|PNG1:C3A C6B|PNG3:C3A C6B|PNSM:C3A C6B|PODG:C4A C4B|POP1:C4A C4B|POP2:C4A C4B|POP3:C4A C4B|POPA:C4A C4B|POPC:C4A C4B|POPE:C4A C4B|POPI:C4A C4B|POPS:C4A C4B|POPX:C4A C4B|POSM:C3A C4B|PPC:C4B|PQPE:C5A C4B|PQPS:C5A C4B|PUDG:D5A C4B|PUPA:D5A C4B|PUPC:D5A C4B|PUPE:D5A C4B|PUPI:D5A C4B|PUPS:D5A C4B|UPC:D5B|XNCE:C5A C6B|XNG1:C5A C6B|XNG3:C5A C6B|XNSM:C5A C6B',
                               start=0,
                               stop=101,
                               )