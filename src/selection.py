import numpy as np
from src.define import RESEXCLUDED

def get_atom_selection(selAtom):

    atomSelection = dict()
    for atomGroup in selAtom.split('|'):
        molName = atomGroup.split(':')[0]
        atomName = atomGroup.split(':')[1].split()
        atomSelection[molName] = atomName

    return atomSelection

def fix_not_selected_mol(top, selAtom):

    for res in top.top.residues:
        if not (res.name in list(selAtom.keys())) and not (res.name in RESEXCLUDED): 
            selAtom[res.name] = []

    return selAtom

def get_index_of_selected_atom(top, atomSelection, fillUnselected=False):

    # Only the first atom in the selection command 
    # will be selected in this function.
    
    atomIdx = []
    for res in top.top.residues:
        if not (res.name in RESEXCLUDED) and len(atomSelection[res.name]):
            for atom in res.atoms_by_name(atomSelection[res.name][0]): 
                atomIdx.append(atom.index)
        else:
            if fillUnselected:
                atomIdx.append(res.atom(0).index)
            else:
                atomIdx.append(np.nan)
        
    return np.array(atomIdx)

def get_selected_residue(top, selAtom):

    selRes = []
    for iRes, res in enumerate(top.top.residues):
        if len(selAtom[res.name]) > 0:
            selRes.append(iRes)    
    
    return selRes