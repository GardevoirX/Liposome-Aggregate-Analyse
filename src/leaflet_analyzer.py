import pickle
import mdtraj as md
from toolkit import read_file, write_file

class Analyzer():

    def __init__(self, fileName, trajFileName, topFileName):

        self.leafletCollection = read_file(fileName)
        self.trajFileName = trajFileName
        self.top = md.load(topFileName)

    def load_traj(self, stride=None):

        self.traj = md.load(self.trajFileName, top=self.top, stride=stride)

        return self.traj

