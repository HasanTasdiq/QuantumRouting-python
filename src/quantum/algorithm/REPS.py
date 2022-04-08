from platform import node
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
        self.numOfSDpair = len(self.srcDstPair)

    def p2(self):
        f, self.ti = self.PFT()
        for i in range(self.numOfSDpair):
            for link in self.topo.links:
                if f[i][(link.n1, link.n2)]:
                    link.assignQubits()
                    f[i][(link.n1, link.n2)] = f[i][(link.n1, link.n2)] - 1

        print('p2 end')
    
    def p4(self):
        self.EPS(self.ti)
        self.ELS()
        print('p4 end') 
    
    # return f_i(u, v)
    def PFT(self):
        
        # initialize f_i(u, v) ans t_i

        fi = [{}] * self.numOfSDpair
        ti = [0] * self.numOfSDpair
        for i in range(self.numOfSDpair):
            fi[i] = {} # erase the reference
            for link in self.topo.links:
                fi[i][(link.n1, link.n2)] = 0
    
        # LP
        # f, t = ...

        print('PFT end')
        return (fi, ti)

    def EPS(self, ti):
        # initialize f_k_i(u, v), t_k_i
        fki = [] * self.numOfSDpair
        tki = [] * self.numOfSDpair

        for i in range(self.numOfSDpair):
            fki[i] = [{}] * ti[i]
            tki[i] = [0] * ti[i]
            for k in range(ti[i]):
                fki[i][k] = {}
                for link in self.topo.link:
                    fki[i][k][(link.n1, link.2)] = 0

        print('EPS end')
    def ELS(self):
        print('ELS end')
if __name__ == '__main__':
    
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99])])