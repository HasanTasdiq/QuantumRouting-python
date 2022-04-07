import sys
from threading import local
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link


class REPS(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.name = "REPS"

    def p2(self):
        self.PFT()
        self.EPS()
        print('p2 end')
    
    def p4(self):
        self.ELS()
        print('p4 end') 
    
    def PFT(self):
        numOfNodes = len(self.topo.nodes)
        F = [0] * numOfNodes
        for i in range(numOfNodes):
            F[i] = {}

        for link in self.topo.links:
            F[i][link] = 0
        # LP
        
        print('PFT end')
			
    def EPS(self):
        # ...
        print('EPS end')
    def ELS(self):
        # ...
        print('ELS end')
if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99])])