import pickle
from xmlrpc.client import boolean
import mdtraj as md
import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from numba import jit
from itertools import combinations
from toolkit import read_file, write_file, setEnv

class Leaflet():
    
    def __init__(self, iFrame, iAgg, location, molIdx, composition=None, nMol=None):
        
        self.iFrame = iFrame
        self.iAgg = iAgg
        self.location = location
        self.molIdx = molIdx
        self.nMol = len(molIdx)
        
    def get_composition(self, molType):
        
        self.composition = {}
        mol = molType[self.molIdx]
        molName = set(molType)
        for name in molName:
            self.composition[name] = np.sum(mol == name)
        self.composition = pd.Series(self.composition)
                
        return self.composition
    
    def get_output(self):
        
        output = str()
        for idx in self.molIdx:
            output += str(idx + 1) + ' '
        print(output)
        
        return output

def main():
    
    topFile, trajFile, trjSkip, allSetsFile, selHeadAtom, selTailAtom, start, stop, skip, distanceCutoff, parallelDegreeCutoff, minLeafletSize, chunkSize, outputPref = parse_args()
    setEnv(4)
    top = md.load(topFile)
    molType, headAtomIdx, head_resIdxMatrix, nHeadAtomPerMol, tail_resIdxMatrix, nTailAtomPerMol, selRes = init(top, selHeadAtom, selTailAtom)
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
    for iChunk, chunk in enumerate(md.iterload(trajFile, chunk=chunkSize, top=top, skip=start)):
        if iChunk > nChunk:
            break
        else:
            allSetsStart, allSetsStop = get_start_and_stop(iChunk, nChunk, chunkSize, nFrames)
        adjMatrix = np.array(generate_adjacency_matrix_of_atom_pairs(chunk, headAtomIdx, allSets[allSetsStart:allSetsStop], distanceCutoff))
        normalVector = get_normal_vector_of_every_molecular(chunk, selRes, head_resIdxMatrix, nHeadAtomPerMol, tail_resIdxMatrix, nTailAtomPerMol, distanceCutoff)
        assignment = assign_leaflet(adjMatrix, normalVector, parallelDegreeCutoff, minLeafletSize)
        leafIdx.append(assign_leaflet_index(assignment, allSets[allSetsStart:allSetsStop], molType))

    write_file(np.concatenate(leafIdx), os.path.join(outputPref, 'leaflet.pickle'))

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='To assign lipid to leaflets for every frames provided')
    parser.add_argument('-top', dest='topFile', help='All the format can be read by mdtraj is supported.')
    parser.add_argument('-trj', dest='trajFile')
    parser.add_argument('-trjSkip', dest='trjSkip', default=1)
    parser.add_argument('-set', dest='allSetsFile', help='The length of the all_sets file must be the same as the trjSkip times traj file\'s. If not provided, every molecule will be considered as in a single aggregate/', default=None)
    parser.add_argument('-head', dest='selHeadAtom', help='The atoms/beads chosen for every kind of molecules should be specified. For example, "-head \'DPPC:NC3 PO4|DPPE:NH3 PO4\'".')
    parser.add_argument('-tail', dest='selTailAtom')
    parser.add_argument('-start', dest='start', help='Frames in traj[start:stop:skip] will be processed. If not provided, the whole trajectory will be processed.', 
                        default=None)
    parser.add_argument('-stop', dest='stop', default=None)
    parser.add_argument('-skip', dest='skip', default=1)
    parser.add_argument('-dist', dest='distanceCutoff', help='Only atoms within this cutoff will be considered for atom pairs and normal vector calculation.', default=1.66)
    parser.add_argument('-degree', dest='parallelDegreeCutoff', help='If the angle of two vectors is smaller than this value, the two vectors will be considered as a pair of parallel vectors.', default=14)
    parser.add_argument('-min', dest='minLeafletSize', help='Only the leaflet containing moleculars more than this value will be considered as a leaflet.',
                        default=30)
    parser.add_argument('-chunk', dest='chunkSize', help='Parameter for mdtraj.iterload()',
                        default=1000)
    parser.add_argument('-o', dest='outputPref', help='The file name of the output file.', default='./')
    args = parser.parse_args()
    write_args(args, args.outputPref)
    return args.topFile, args.trajFile, int(args.trjSkip), args.allSetsFile, get_atom_selection(args.selHeadAtom), get_atom_selection(args.selTailAtom), identify_num(args.start), identify_num(args.stop), int(args.skip), float(args.distanceCutoff), float(args.parallelDegreeCutoff), int(args.minLeafletSize), int(args.chunkSize), args.outputPref

def write_args(args, outputPref):

    write_file(args, os.path.join(outputPref, 'leaflet_args.pickle'))

def identify_num(num):

    if isinstance(num, str): return int(num)

def get_atom_selection(selAtom):

    atomSelection = dict()
    for atomGroup in selAtom.split('|'):
        molName = atomGroup.split(':')[0]
        atomName = atomGroup.split(':')[1].split()
        atomSelection[molName] = atomName

    return atomSelection

def init(top, selHeadAtom, selTailAtom):

    molType = get_mol_type(top)
    selHeadAtom = fix_not_selected_mol(top, selHeadAtom)
    selTailAtom = fix_not_selected_mol(top, selTailAtom)
    headAtomIdx = get_index_of_selected_atom(top, selHeadAtom)
    head_resIdxMatrix, nHeadAtomPerMol = get_atom_residue_matrix(top, selHeadAtom, molType)
    tail_resIdxMatrix, nTailAtomPerMol = get_atom_residue_matrix(top, selTailAtom, molType)
    selRes = get_selected_residue(top, selHeadAtom)

    return molType, headAtomIdx, head_resIdxMatrix, nHeadAtomPerMol, tail_resIdxMatrix, nTailAtomPerMol, selRes

def fix_not_selected_mol(top, selAtom):

    for residue in top.top.residues:
        if not (residue.name in list(selAtom.keys())): selAtom[residue.name] = []

    return selAtom

def get_index_of_selected_atom(trajectory, atomSelection):
    
    atomIdx = []
    for res in trajectory.top.residues:
        if len(atomSelection[res.name]):
            for atom in res.atoms_by_name(atomSelection[res.name][0]): 
                atomIdx.append(atom.index)
        else:
            atomIdx.append(np.nan)
        
    return np.array(atomIdx)

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

def generate_adjacency_matrix_of_atom_pairs(trajectory, headAtomIdx, allSets, cutoff):
    
    adjMatrix = []
    for iFrame in range(len(allSets)):
        # Assuming that a molecular cannot transfer from one aggregate to another unless two aggregates fuse
        print('Constructing the adjacency matrix of Frame {}'.format(iFrame))
        if not(iFrame) or (len(allSets[iFrame]) != len(allSets[iFrame - 1])):
            molPairs = np.concatenate([cartesian_product(agg) for agg in allSets[iFrame]])
            atomPairs = headAtomIdx[molPairs]
        distances = md.compute_distances(trajectory[iFrame], atomPairs) < cutoff
        adjMatrix.append(construct_adjacency_matrix(molPairs[distances[0]], len(headAtomIdx)))

    return adjMatrix

def get_normal_vector_of_every_molecular(trajectory, selRes, head_resIdxMatrix, nHeadAtomPerMol, tail_resIdxMatrix, nTailAtomPerMol, cutoff):

    import open3d as o3d

    normalVector = {}
    headPos = head_resIdxMatrix.dot(trajectory.xyz)
    headPos = headPos * np.array([nHeadAtomPerMol]).T
    tailPos = tail_resIdxMatrix.dot(trajectory.xyz)
    tailPos = tailPos * np.array([nTailAtomPerMol]).T
    molecularOrientation = tailPos - headPos
    print(headPos)
    print(nHeadAtomPerMol)
    for iFrame in range(len(trajectory)):
        print('Computing the normal vector of Frame {}'.format(iFrame))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(headPos[::, iFrame, ::])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=cutoff, max_nn=20))
        pcd.normalize_normals()
        normals = np.asarray(pcd.normals)
        toConvert = np.argwhere((product(molecularOrientation[::, iFrame, ::], normals)) < 0)
        normals[toConvert] = -normals[toConvert] 
        normalVector[iFrame] = normals
        print(len(normals))
    return normalVector

def assign_leaflet(adjMatrix, normalVector, degreeCutoff, minLeafletSize):
    
    leafletAssignment = []
    parallelDegree = np.cos(np.deg2rad(degreeCutoff))
    for iFrame in range(len(adjMatrix)):
        pairs = np.argwhere(np.triu(adjMatrix[iFrame]) == True)
        productOfVector = product(normalVector[iFrame][pairs[::, 0]], normalVector[iFrame][pairs[::, 1]])
        pairsNotParallel = (productOfVector < parallelDegree)
        for (iMol, jMol) in (pairs[pairsNotParallel]):
            adjMatrix[iFrame, iMol, jMol] = adjMatrix[iFrame, jMol, iMol] = False
        G = nx.from_numpy_matrix(adjMatrix[iFrame])
        assign = []
        for g in nx.connected_components(G):
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
    print(selRes)
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
    atom_resIdxMatrix = np.array([np.isin(ref, np.array([traj.top.residue(iRes).atom(atomName).index 
                                                         for atomName in selAtom[molType[iRes]]])) 
                                  for iRes in range(traj.top.n_residues)])
    nAtomPerMol = []
    for molName in molType:
        if len(selAtom[molName]):
            nAtomPerMol.append(1 / len(selAtom[molName]))
        else:
            nAtomPerMol.append(np.nan)
    nAtomPerMol = np.array(nAtomPerMol)
    
    return atom_resIdxMatrix, nAtomPerMol

def get_selected_residue(top, selAtom):

    selRes = []
    for iRes, res in enumerate(top.top.residues):
        if len(selAtom[res.name]) > 0:
            selRes.append(iRes)    
    
    return selRes

@jit(nopython=True)
def product(v1, v2):
    
    return np.sum(v1 * v2, axis=1)

@jit(nopython=True)
def construct_adjacency_matrix(pair, nMol):
    
    adjMatrix = np.full((nMol, nMol), False)
    for (i, j) in pair:
        adjMatrix[i, j] = True
        adjMatrix[j, i] = True
        
    return adjMatrix

if __name__ == "__main__":
    main()