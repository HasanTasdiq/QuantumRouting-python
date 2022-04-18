import sys
import math
from threading import local
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
import gurobipy as gp
from gurobipy import quicksum


class REPS(AlgorithmBase):
    def __init__(self, topo):
        super().__init__(topo)
        self.name = "REPS"

    def p2(self):
        self.PFT() # compute (self.t_i, self.f_i)
        for SDpair in self.srcDstPairs:
            for edge in self.topo.edges:
                n1 = edge[0]
                n2 = edge[1]
                if self.f_i[SDpair][edge]:
                    assignCount = 0
                    for link in n1.links:
                        if link.contains(n2) and link.assignable():
                            # link(n1, n2) for u, v in edgeIndices)
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

        self.f_i_LP = {SDpair : {edge : 0 for edge in self.topo.edges} for SDpair in self.srcDstPairs}
        self.t_i_LP = {SDpair : 0 for SDpair in self.srcDstPairs}
        
        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)

        edgeIndices = []

        for edge in self.topo.edges:
            edgeIndices.append((edge[0].id, edge[1].id))
        
        # LP

        m = gp.Model('REPS')
        f = m.addVars(numOfSDpairs, numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS)
        t = m.addVars(numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS)
        x = m.addVars(numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS)
        m.update()
        
        m.setObjective(quicksum(t[i] for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            m.addConstr(quicksum(f[i, s, v] for v in range(numOfNodes)) - quicksum(f[i, v, s] for v in range(numOfNodes)) == t[i])

            d = self.srcDstPairs[i][1].id
            m.addConstr(quicksum(f[i, d, v] for v in range(numOfNodes)) - quicksum(f[i, v, d] for v in range(numOfNodes)) == -t[i])

            for u in range(numOfNodes):
                if u not in [s, d]:
                    m.addConstr(quicksum(f[i, u, v] for v in range(numOfNodes)) - quicksum(f[i, v, u] for v in range(numOfNodes)) == 0)


        
        for (u, v) in edgeIndices:
            dis = self.topo.distance(self.topo.nodes[u].loc, self.topo.nodes[v].loc)
            probability = math.exp(-self.topo.alpha * dis)
            m.addConstr(quicksum(f[i, u, v] + f[i, v, u] for i in range(numOfSDpairs)) <= probability * x[u, v])

            capacity = 0
            for link in self.topo.nodes[v].links:
                if link.contains(self.topo.nodes[u]):
                    capacity += 1
            m.addConstr(x[u, v] <= capacity)
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    m.addConstr(x[u, v] == 0)                
                    for i in range(numOfSDpairs):
                        m.addConstr(f[i, u, v] == 0)

        for u in range(numOfNodes):
            edgeContainu = []
            for (n1, n2) in edgeIndices:
                if u in (n1, n2):
                    edgeContainu.append((n1, n2))
            m.addConstr(quicksum(x[n1, n2] for (n1, n2) in edgeContainu) <= self.topo.nodes[u].remainingQubits)

        m.optimize()
        vars = m.getVars()
        print(m.objVal)
        # for var in vars:
        #     print(var.varName)
        # f, t = ...
        # self.f_i_LP = 
        print('PFT end')
        
    def EPS(self):
        # initialize f_ki(u, v), t_ki
        self.f_ki = {SDpair : {k : {edge : 0 for edge in self.topp.edges} for k in range(self.t_i[SDpair])} for SDpair in self.srcDstPairs}
        self.t_ki = {SDpair : {k : 0 for k in range(self.t_i[SDpair])} for SDpair in self.srcDstPairs}
        
        # LP
        # f, t = ...
        print('EPS end')

    def ELS(self):
        print('ELS end')
if __name__ == '__main__':
    
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99])], 100)