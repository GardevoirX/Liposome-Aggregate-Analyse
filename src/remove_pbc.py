import MDAnalysis as mda
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import track
import sys
sys.path.append('/share/home/qjxu/scripts/aggregate_analyse')
from src.toolkit import setEnv

class PBCRemover():

    def __init__(self, fileName, topFile, ndxFile, outputFileName):

        setEnv(4)
        assert fileName != outputFileName, 'Your output file name is the name of your input trajectory!'
        self.fileName = fileName
        self.topFile = topFile
        self.ndxFile = ndxFile
        self.selectedIdx = self.read_ndx()
        self.outputFileName = outputFileName
        self.u = mda.Universe(self.topFile, self.fileName, in_memory=True)
        self.selectedAtomIdx = self.u.residues[self.selectedIdx].atoms.indices
        self.trajLen = len(self.u.trajectory)

    def run(self):

        CONVERGE = 1
        MAXITER = 50
        loss = self.Loss()
        bias = torch.rand(3, requires_grad=True)
        trainer = torch.optim.SGD([bias], lr=0.1)
        scheduler = CosineAnnealingLR(trainer, 10)
        
        for iFrame in track(range(self.trajLen)):
            iRound = 0
            lastLoss = 0
            self.boxXYZ, (self.a, self.b, self.c), self.boxCenter = self.get_box_info(iFrame)
            allAtomPosition = torch.clone(torch.tensor(self.u.trajectory[iFrame].positions, dtype=float))
            selectedAtomPosition = allAtomPosition[self.selectedAtomIdx]
            while iRound < MAXITER:
                l = loss(selectedAtomPosition + bias, self.boxXYZ, self.boxCenter, self.a, self.b, self.c)
                trainer.zero_grad()
                l.backward()
                trainer.step()
                scheduler.step()
                print(f'Frame {iFrame}    loss: {l.detach().numpy():.2f}    bias: {bias}')
                if abs(l - lastLoss) < CONVERGE:
                    break
                lastLoss = l
            self.update_position(
                    self.move_to_unit_cell(
                            allAtomPosition + bias, self.boxXYZ,
                            self.a, self.b, self.c).detach().numpy()
                    )(self.u.trajectory[iFrame])
        with mda.Writer(self.outputFileName, self.u.trajectory.n_atoms) as W:
            system = self.u.select_atoms('all')
            for ts in self.u.trajectory:
                W.write(system)

        return

    def get_box_info(self, iFrame):

        boxXYZ = torch.tensor([self.u.trajectory[iFrame].triclinic_dimensions[i, i] for i in range(3)])
        a = torch.tensor(self.u.trajectory[iFrame].triclinic_dimensions[0])
        b = torch.tensor(self.u.trajectory[iFrame].triclinic_dimensions[1])
        c = torch.tensor(self.u.trajectory[iFrame].triclinic_dimensions[2])
        boxCenter = boxXYZ / 2
        
        return boxXYZ, (a, b, c), boxCenter

    def read_ndx(self):

        with open(self.ndxFile, 'r') as rfl:
            idx = rfl.readlines()[-1]
            selectedIdx = []
            selectedIdx = np.array(idx.split(), dtype=int)

        return selectedIdx

    def move_to_unit_cell(self, position, boxXYZ, a, b, c):

        # Based on GROMACS Manual 2023 rc1 Equation 5.19

        position -= c * torch.floor(position[::, 2] / boxXYZ[2]).unsqueeze(0).T 
        position -= b * torch.floor(position[::, 1] / boxXYZ[1]).unsqueeze(0).T 
        position -= a * torch.floor(position[::, 0] / boxXYZ[0]).unsqueeze(0).T 

        return position

    def get_position_vector(self, position, boxCenter, boxXYZ, a, b, c):

        # From GROMACS Manual 2023 rc1 Equation 5.19

        r = position - boxCenter
        r -= c * torch.round(r[::, 2] / boxXYZ[2]).unsqueeze(0).T
        r -= b * torch.round(r[::, 1] / boxXYZ[1]).unsqueeze(0).T
        r -= a * torch.round(r[::, 0] / boxXYZ[0]).unsqueeze(0).T

        return r

    def update_position(self, position):

        def wrapped(ts):
            ts.positions = position
            return ts

        return wrapped

    class Loss(nn.Module):

        def __init__(self, multiplier=5.0, **kwargs):

            super(PBCRemover.Loss, self).__init__(**kwargs)
            self.multiplier = multiplier

        def forward(self, position, boxXYZ, boxCenter, a, b, c):

            origin = torch.zeros(3, dtype=float)
            position = PBCRemover.move_to_unit_cell(PBCRemover, position, boxXYZ, a, b, c)
            r = PBCRemover.get_position_vector(PBCRemover, position, boxCenter, boxXYZ, a, b, c)

            return self.multiplier * torch.dist(r, origin)
