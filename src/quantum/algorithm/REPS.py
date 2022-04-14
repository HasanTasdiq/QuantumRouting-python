from platform import node
import sys
from threading import local
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
import gurobipy as gp


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

        self.f_i = {SDpair : {edge : 0 for edge in self.topo.edges} for SDpair in self.srcDstPairs}
        self.t_i = {SDpair : 0 for SDpair in self.srcDstPairs}
        
        numOfNodes = len(self.topo.nodes)
        
        nodeIndices = []
        edgeIndices = []

        for i in range(numOfNodes):
            nodeIndices.append(i)

        for edge in self.topo.edges:
            edgeIndices.append(edge.n1, edge.n2)
        
        # LP
        ub_t = {}

        m = gp.Model('REPS')
        f = m.addVars(numOfNodes, numOfNodes)
        t = m.addVars(numOfNodes)
        x = m.addVars(numOfNodes, numOfNodes)
        # f, t = ...

        print('PFT end')
        
    def EPS(self):
        # initialize f_ki(u, v), t_ki
        self.f_ki = {SDpair : {k : {edge : 0 for edge in self.topp.edges} for k in self.ti[SDpair]} for SDpair in self.srcDstPairs}
        self.t_ki = {SDpair : {k : 0 for k in range(self.ti[SDpair])} for SDpair in self.srcDstPairs}
        
        # LP
        # f, t = ...
        print('EPS end')

    def ELS(self):
        print('ELS end')
if __name__ == '__main__':
    
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99])])