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

    def p2(self):
        self.PFT() # compute (self.t_i, self.f_i)
        for SDpair in self.srcDstPairs:
            for edge in self.topo.edges:
                n1 = edge.n1
                n2 = edge.n2
                if self.f_i[SDpair][edge]:
                    assignCount = 0
                    for link in n1.links:
                        if link.contains(n2) and link.assignable():
                            # link(n1, n2)
                            link.assign()

                            assignCount += 1
                            if assignCount == self.f[SDpair][edge]:
                                break
        print('p2 end')
    
    def p4(self):
        self.EPS()
        self.ELS()
        print('p4 end') 
    
    # return f_i(u, v)
    def PFT(self):
        
        # initialize f_i(u, v) ans t_i

        self.f_i = {i : {uv : 0 for uv in self.topo.edges} for i in self.srcDstPairs}
        self.t_i = {i : 0 for i in self.srcDstPairs}

        # LP
        # f, t = ...

        print('PFT end')
        
    def EPS(self):
        # initialize f_ki(u, v), t_ki
        f_ki = {i : {k : {uv : 0 for uv in self.topp.edges} for k in self.ti[i]} for i in self.srcDstPairs}
        t_ki = {i : {k : 0 for k in self.ti[i]} for i in self.srcDstPairs}
        
        # LP
        # f, t = ...
        print('EPS end')

        return (f_ki, t_ki)
    def ELS(self):
        print('ELS end')
if __name__ == '__main__':
    
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99])])