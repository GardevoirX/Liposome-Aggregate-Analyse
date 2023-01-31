from xmlrpc.client import boolean
import os
import sys
sys.path.append('.')

import argparse
import mdtraj as md
import numpy as np
import cupy as cp
import open3d as o3d
import pandas as pd
import networkx as nx
from numba import jit
from rich.progress import track
from itertools import combinations
from src.leaflet import Leaflet
from src.toolkit import read_file, write_file, setEnv
from src.selection import *


def main():
    
    topFile, trajFile, trjSkip, allSetsFile, selHeadAtom, selTailAtom, \
            start, stop, skip, distanceCutoff, parallelDegreeCutoff, \
            minLeafletSize, chunkSize, outputPref = parse_args()
    setEnv(4)
    top = md.load(topFile)
    molType, headAtomIdx, head_resIdxMatrix, nHeadAtomPerMol, \
            tail_resIdxMatrix, nTailAtomPerMol, selRes \
            = init(top, selHeadAtom, selTailAtom)
    assert len(headAtomIdx) == top.n_residues
    print('Initialization completed!')

    if isinstance(allSetsFile, str):
        allSets = read_allSets(allSetsFile, start, stop, skip, trjSkip)
    else:
        allSets = generate_allSets(selRes, start, stop)

    nFrames = len(allSets)
    print('All sets read! Totally {} frames!'.format(nFrames))

    nChunk = int(nFrames / chunkSize)
    leafIdx = []
    for iChunk, chunk in track(enumerate(md.iterload(trajFile, chunk=chunkSize, top=top, skip=start))):
        if iChunk > nChunk:
            break
        else:
            allSetsStart, allSetsStop = get_start_and_stop(iChunk, nChunk, 
                                                           chunkSize, nFrames)
        aggGraph = generate_adjacency_matrix_of_atom_pairs(
                chunk, headAtomIdx, 
                allSets[allSetsStart:allSetsStop], distanceCutoff)
        normalVector = get_normal_vector_of_every_molecular(
                chunk, selRes, head_resIdxMatrix, nHeadAtomPerMol, 
                tail_resIdxMatrix, nTailAtomPerMol, distanceCutoff)
        assignment = assign_leaflet(
                aggGraph, normalVector, parallelDegreeCutoff, minLeafletSize)
        leafIdx.append(assign_leaflet_index(
                assignment, allSets[allSetsStart:allSetsStop], molType))

    write_file(np.concatenate(leafIdx), 
            os.path.join(outputPref, 'leaflet.pickle'))

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='To assign lipid to leaflets for every frames provided')
    parser.add_argument('-top', dest='topFile', help='All the format can be read by mdtraj is supported.')
    parser.add_argument('-trj', dest='trajFile')
    parser.add_argument('-trjSkip', dest='trjSkip', default=1, type=int)
    parser.add_argument('-set', dest='allSetsFile', help='The length of the all_sets file must be the same as the trjSkip times traj file\'s. If not provided, every molecule will be considered as in a single aggregate/', default=None)
    parser.add_argument('-head', dest='selHeadAtom', help='The atoms/beads chosen for every kind of molecules should be specified. For example, "-head \'DPPC:NC3 PO4|DPPE:NH3 PO4\'".')
    parser.add_argument('-tail', dest='selTailAtom')
    parser.add_argument('-start', dest='start', help='Frames in traj[start:stop:skip] will be processed. If not provided, the whole trajectory will be processed.', 
                        default=None)
    parser.add_argument('-stop', dest='stop', default=None)
    parser.add_argument('-skip', dest='skip', default=1, type=int)
    parser.add_argument('-dist', dest='distanceCutoff', help='Only atoms within this cutoff will be considered for atom pairs and normal vector calculation.', default=1.66, type=float)
    parser.add_argument('-degree', dest='parallelDegreeCutoff', help='If the angle of two vectors is smaller than this value, the two vectors will be considered as a pair of parallel vectors.', default=14, type=float)
    parser.add_argument('-min', dest='minLeafletSize', help='Only the leaflet containing moleculars more than this value will be considered as a leaflet.',
                        default=30, type=float)
    parser.add_argument('-chunk', dest='chunkSize', help='Parameter for mdtraj.iterload()',
                        default=1000, type=int)
    parser.add_argument('-o', dest='outputPref', help='The file name of the output file.', default='./')
    args = parser.parse_args()
    write_args(args, args.outputPref)
    return args.topFile, args.trajFile, args.trjSkip, args.allSetsFile,\
            get_atom_selection(args.selHeadAtom), \
            get_atom_selection(args.selTailAtom), \
            identify_num(args.start), identify_num(args.stop), args.skip,\
            args.distanceCutoff, args.parallelDegreeCutoff, \
            args.minLeafletSize, args.chunkSize, args.outputPref

def write_args(args, outputPref):

    write_file(args, os.path.join(outputPref, 'leaflet_args.pickle'))

def identify_num(num):

    if isinstance(num, str): return int(num)

def init(top, selHeadAtom, selTailAtom):

    molType = get_mol_type(top)
    selHeadAtom = fix_not_selected_mol(top, selHeadAtom)
    selTailAtom = fix_not_selected_mol(top, selTailAtom)
    headAtomIdx = get_index_of_selected_atom(top, selHeadAtom)
    head_resIdxMatrix, nHeadAtomPerMol \
        = get_atom_residue_matrix(top, selHeadAtom, molType)
    tail_resIdxMatrix, nTailAtomPerMol = \
        get_atom_residue_matrix(top, selTailAtom, molType)
    selRes = get_selected_residue(top, selHeadAtom)

    return molType, headAtomIdx, head_resIdxMatrix, nHeadAtomPerMol, tail_resIdxMatrix, nTailAtomPerMol, selRes

def cartesian_product(array):

    if len(array) == 1:
        return [(array[0], array[0])]
    else:
        return list(combinations(array, 2))

def get_mol_type(traj):
    
    return np.array([mol.name for mol in traj.top.residues])

def get_start_and_stop(iChunk, nChunk, chunkSize, allSetsLen):

    allSetsStart = chunkSize * iChunk
    if iChunk == nChunk:
        allSetsStop = allSetsLen
    else:
        allSetsStop = chunkSize * (iChunk + 1)
    
    return allSetsStart, allSetsStop

def generate_adjacency_matrix_of_atom_pairs(trajectory, headAtomIdx, \
        allSets, cutoff):
    
    nSelectedMol = len(headAtomIdx)
    trajLen = len(allSets)
    aggGraph = []
    lFrame = 0
    for iFrame in range(trajLen):
        # Assuming that a molecular cannot transfer from one aggregate to another unless two aggregates fuse
        if iFrame == 0:
            molPairs = np.concatenate([
                cartesian_product(agg) for agg in allSets[iFrame]])
            atomPairs = headAtomIdx[molPairs]
            lFrame = iFrame
            
        if (iFrame != 0) and (len(allSets[iFrame]) != len(allSets[iFrame - 1])):
            # pairs need change
            print(f'Computing the distance matrix of Frame {lFrame} - {iFrame}')
            distances = md.compute_distances(
                    trajectory[lFrame:iFrame], atomPairs) < cutoff
            print(f'Constructing the adjacency matrix of Frame {lFrame} - {iFrame}')
            for distance in distances:
                G = nx.Graph()
                G.add_nodes_from(range(nSelectedMol))
                nx.from_edgelist(molPairs[distance], G)
                aggGraph.append(G)
            molPairs = np.concatenate([
                cartesian_product(agg) for agg in allSets[iFrame]])
            atomPairs = headAtomIdx[molPairs]
            lFrame = iFrame

        if iFrame == (trajLen - 1):
            # the last frame needs special treatment
            pass
        else:
            continue

        print(f'Computing the distance matrix of Frame {lFrame} - {iFrame}')
        distances = md.compute_distances(
            trajectory[lFrame:iFrame + 1],atomPairs) < cutoff
        print(f'Constructing the adjacency matrix of Frame {lFrame} - {iFrame}')
        for distance in distances:
            G = nx.Graph()
            G.add_nodes_from(range(nSelectedMol))
            nx.from_edgelist(molPairs[distance], G)
            aggGraph.append(G)

    return aggGraph

def get_normal_vector_of_every_molecular(trajectory, selRes, \
            head_resIdxMatrix, nHeadAtomPerMol, tail_resIdxMatrix, \
            nTailAtomPerMol, cutoff):

    normalVector = {}
    molecularOrientation, headPos = get_mol_orientation_CUDA(trajectory.xyz, 
            head_resIdxMatrix, tail_resIdxMatrix, 
            nHeadAtomPerMol, nTailAtomPerMol)
    for iFrame in range(len(trajectory)):
        print('Computing the normal vector of Frame {}'.format(iFrame))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(headPos[iFrame])
        pcd.estimate_normals(search_param=
                o3d.geometry.KDTreeSearchParamHybrid(radius=cutoff, max_nn=20))
        pcd.normalize_normals()
        normals = np.asarray(pcd.normals)
        toConvert = np.argwhere((product(molecularOrientation[iFrame], normals)) < 0)
        normals[toConvert] = -normals[toConvert] 
        normalVector[iFrame] = normals

    return normalVector

def get_mol_orientation(xyz, head_resIdxMatrix, tail_resIdxMatrix, nHeadAtomPerMol, nTailAtomPerMol):


    print('Now getting the molecular orientation.')
    headPos = np.array([dot_jit(np.float32(head_resIdxMatrix), frame) for frame in xyz])
    headPos = np.array([hadamard_product(frame, nHeadAtomPerMol) for frame in headPos])
    tailPos = np.array([dot_jit(np.float32(tail_resIdxMatrix), frame) for frame in xyz])
    tailPos = np.array([hadamard_product(frame, nTailAtomPerMol) for frame in tailPos])
    molecularOrientation = tailPos - headPos

    return molecularOrientation, headPos

def get_mol_orientation_CUDA(xyz, head_resIdxMatrix, tail_resIdxMatrix, nHeadAtomPerMol, nTailAtomPerMol):


    xyz = cp.array(xyz)
    head_resIdxMatrix = cp.array(head_resIdxMatrix)
    tail_resIdxMatrix = cp.array(tail_resIdxMatrix)
    nHeadAtomPerMol = cp.array(nHeadAtomPerMol)
    nTailAtomPerMol = cp.array(nTailAtomPerMol)
    headPos = cp.dot(head_resIdxMatrix, xyz).transpose((1, 0, 2))
    headPos = cp.multiply(headPos, nHeadAtomPerMol)
    tailPos = cp.dot(tail_resIdxMatrix, xyz).transpose((1, 0, 2))
    tailPos = cp.multiply(tailPos, nTailAtomPerMol)
    molecularOrientation = cp.subtract(tailPos, headPos)

    return cp.asnumpy(molecularOrientation), cp.asnumpy(headPos)

def assign_leaflet(aggGraph, normalVector, degreeCutoff, minLeafletSize):
    
    leafletAssignment = []
    parallelDegree = np.cos(np.deg2rad(degreeCutoff))
    for iFrame in range(len(aggGraph)):
        pairs = np.array(aggGraph[iFrame].edges())
        productOfVector = product(normalVector[iFrame][pairs[::, 0]], normalVector[iFrame][pairs[::, 1]])
        pairsNotParallel = (productOfVector < parallelDegree)
        aggGraph[iFrame].remove_edges_from(pairs[pairsNotParallel])
        assign = []
        for g in nx.connected_components(aggGraph[iFrame]):
            if len(g) > minLeafletSize: 
                assign.append(np.array(list(g)))
        leafletAssignment.append(assign)

    return leafletAssignment

def read_allSets(fileName, start, stop, skip, trjSkip):
    
    if not (start is None):
        start = start * trjSkip
    if not (stop is None):
        stop = stop * trjSkip

    f = open(fileName, 'r')
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

def generate_allSets(selRes, start, stop):

    selRes = np.array(selRes)
    allSets = np.array([[selRes] for _ in range(start, stop)])

    return allSets

def assign_leaflet_index(assignment, allSets, molType):

    leafletIdx = []
    for iFrame in range(len(assignment)):
        leafletIdx.append({})
        leafletsInAgg = np.full(len(allSets[iFrame]), 1)
        for leaflet in assignment[iFrame]:
            for iAgg in range(len(allSets[iFrame])):
                if leaflet[0] in allSets[iFrame][iAgg]:
                        leafletIdx[iFrame][(iAgg, leafletsInAgg[iAgg])] = Leaflet(iFrame, iAgg, leafletsInAgg[iAgg], leaflet)
                        leafletIdx[iFrame][(iAgg, leafletsInAgg[iAgg])].get_composition(molType)
                        leafletsInAgg[iAgg] += 1

    return leafletIdx

def get_atom_residue_matrix(traj, selAtom, molType):
    
    ref = np.array(range(traj.top.n_atoms))
    atom_resIdxMatrix = np.array(
        [np.isin(ref, 
            np.array([traj.top.residue(iRes).atom(atomName).index 
                        for atomName in selAtom[molType[iRes]]])) 
                      for iRes in range(traj.top.n_residues)])
    nAtomPerMol = []
    for molName in molType:
        if len(selAtom[molName]):
            nAtomPerMol.append(1 / len(selAtom[molName]))
        else:
            nAtomPerMol.append(np.nan)
    nAtomPerMol = np.array([nAtomPerMol]).T
    
    return atom_resIdxMatrix, nAtomPerMol

@jit(nopython=True)
def product(v1, v2):
    
    return np.sum(v1 * v2, axis=1)

@jit(fastmath=True, nopython=True)
def dot_jit(a, b):
    
    return a.dot(b) 

@jit(fastmath=True, nopython=True)
def hadamard_product(a, b):

    return a * b

@jit(nopython=True)
def construct_adjacency_matrix(pair, nMol):
    
    adjMatrix = np.full((nMol, nMol), False)
    for (i, j) in pair:
        adjMatrix[i, j] = True
        adjMatrix[j, i] = True
        
    return adjMatrix

if __name__ == "__main__":
    main()