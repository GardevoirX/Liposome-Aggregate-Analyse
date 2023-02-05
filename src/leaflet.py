import numpy as np
import pandas as pd

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

    def add_new_mol(self, molIdx):

        self.molIdx = np.concatenate([self.molIdx, molIdx])
        self.nMol += len(molIdx)