import os
import sys
import argparse


import numpy as np
import pandas as pd
import cupy as cp
import open3d as o3d
import mdtraj as md
import networkx as nx
import MDAnalysis as mda
from MDAnalysis.analysis.distances import self_distance_array
from numba import jit
from rich.progress import track
from itertools import combinations
from math import ceil
from src.leaflet import Leaflet
from src.io import read_file, write_file
from src.toolkit import setEnv
from src.selection import *


class LeafletAssigner():

    def __init__(self, topFile, trajFile, \
                 selHeadAtom, selTailAtom, allSetsFile=None, \
                 start=None, stop=None, skip=None, trjSkip=1,\
                 distanceCutoff=16.6, parallelDegreeCutoff=4, \
                 minLeafletSize=30, chunkSize=100, outputPref='./'):

        self.topFile = topFile
        self.trajFile = trajFile
        self.top = mda.Universe(topFile, trajFile)
        self.nRes = len(self.top.residues)
        self.traj = self.top.trajectory
        self.trajLen = len(self.traj)
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
        self.nChunk = round(self.nFrames / chunkSize)
        self.outputPref = outputPref

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

        self.leafIdx = np.concatenate(self.leafIdx)

        return self.leafIdx           

    def generate_adjacency_matrix_of_atom_pairs(self, start, stop):
        
        nSelectedMol = len(self.headAtomIdx)
        aggGraph = []
        dist = []
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
            dist.append(np.concatenate(pairDistances))
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
    
        return np.array([mol.resname for mol in top.residues])
    
    @staticmethod
    def _get_atom_residue_matrix(traj, selAtom, molType):
    
        ref = np.array(range(len(traj.atoms)))
        atom_resIdxMatrix = np.array(
            [np.isin(ref, 
                np.array([atom.index 
                          for atom in res.atoms if atom.name in selAtom[res.resname]]))
             for res in traj.residues])
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
    
