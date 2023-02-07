import numpy as np
import pandas as pd

class XVGReader():
    
    def __init__(self, fileName):
        
        self.fileName = fileName
        self.data = pd.DataFrame()
            
    def read(self):
        
        with open(self.fileName, 'r') as rfl:
            content = rfl.readlines()
        self.legend = {}
        tempData = {}
        for line in content:
            if line[0] == '#':
                continue
            elif line[0] == '@':
                split = line.split()
                if split[1] == 'title':
                    self.title = line.split('"')[1]
                elif split[1] == 'xaxis':
                    self.xaxis = line.split('"')[1]
                    tempData[0] = []
                elif split[1] == 'yaxis':
                    self.yaxis = line.split('"')[1]
                elif len(split) > 2 and split[2] == 'legend':
                    self.legend[split[1]] = line.split('"')[1]
                    tempData[len(tempData.keys())] = []
            else:
                for i, data in enumerate(line.split()):
                    tempData[i].append(data)
        
        self.data[self.xaxis] = np.array(tempData[0], dtype=float)
        for i, key in enumerate(self.legend.keys()):
            self.data[self.legend[key]] = np.array(tempData[i+1], dtype=float)
            
        return self.data
