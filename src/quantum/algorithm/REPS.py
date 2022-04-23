from gettext import find
from operator import ne
import sys
import math
from threading import local
from tkinter.messagebox import NO
from turtle import width
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

    def getVarName(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')

    def p2(self):
        self.PFT() # compute (self.t_i, self.f_i)
        for SDpair in self.srcDstPairs:
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                need = self.f_i[SDpair][(u, v)] + self.f_i[SDpair][(u, v)]
                if self.f_i[SDpair][edge]:
                    assignCount = 0
                    for link in u.links:
                        if link.contains(v) and link.assignable():
                            # link(u, v) for u, v in edgeIndices)
                            assignCount += 1
                            if assignCount == need:
                                break
        print('p2 end')
    
    def p4(self):
        self.EPS()
        self.ELS()
        print('p4 end') 
    
    # return f_i(u, v)

    def LP1(self):
        # initialize f_i(u, v) ans t_i

        self.f_i_LP = {SDpair : {} for SDpair in self.srcDstPairs}
        self.t_i_LP = {SDpair : 0 for SDpair in self.srcDstPairs}
        
        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)

        edgeIndices = []
        notEdge = []
        for edge in self.topo.edges:
            edgeIndices.append((edge[0].id, edge[1].id))
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))
        # LP

        m = gp.Model('REPS')
        f = m.addVars(numOfSDpairs, numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "f")
        t = m.addVars(numOfSDpairs, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "t")
        x = m.addVars(numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "x")
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
        
        for (u, v) in notEdge:
            m.addConstr(x[u, v] == 0)                
            for i in range(numOfSDpairs):
                m.addConstr(f[i, u, v] == 0)

        for u in range(numOfNodes):
            edgeContainu = []
            for (n1, n2) in edgeIndices:
                if u in (n1, n2):
                    edgeContainu.append((n1, n2))
                    edgeContainu.append((n2, n1))
            m.addConstr(quicksum(x[n1, n2] for (n1, n2) in edgeContainu) <= self.topo.nodes[u].remainingQubits)

        m.optimize()

        for i in range(numOfSDpairs):
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                varName = self.getVarName('f', [i, u.id, v.id])
                SDpair = self.srcDstPairs[i]
                self.f_i_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for edge in self.topo.edges:
                u = edge[1]
                v = edge[0]
                varName = self.getVarName('f', [i, u.id, v.id])
                SDpair = self.srcDstPairs[i]
                self.f_i_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for (u, v) in notEdge:
                u = self.topo.nodes[u]
                v = self.topo.nodes[v]
                self.f_i_LP[SDpair][(u, v)] = 0
            
            
            varName = self.getVarName('t', [i])
            self.t_i_LP = m.getVarByName(varName).x

    def PFT(self):

        self.f_i = {SDpair : {} for SDpair in self.srcDstPairs}
        self.t_i = {SDpair : 0 for SDpair in self.srcDstPairs}

        self.f_i = {SDpair : {} for SDpair in self.srcDstPairs}
        self.t_i = {SDpair : 0 for SDpair in self.srcDstPairs}

        self.LP1()
        failedFindPath = False
        while not failedFindPath:
            failedFindPath = True
            P_i = {}
            for SDpair in self.srcDstPairs:
                P_i[SDpair] = self.findPaths(SDpair)

            for SDpair in self.srcDstPairs:
                K = len(P_i[SDpair])
                if K >= 1:
                    failedFindPath = False

                for k in range(K):
                    width = self.pathWidth(P_i[SDpair][k], SDpair)

        print('PFT end')
        
    def EPS(self):
        # initialize f_ki(u, v), t_ki
        self.f_ki = {SDpair : {k : {} for k in range(self.t_i[SDpair])} for SDpair in self.srcDstPairs}
        self.t_ki = {SDpair : {k : 0 for k in range(self.t_i[SDpair])} for SDpair in self.srcDstPairs}
        
        # LP
        # f, t = ...
        print('EPS end')

    def ELS(self):
        print('ELS end')


    def findPaths(self, SDpair: int):
        src = SDpair[0]
        dst = SDpair[1]
        self.pathList = []
        self.currentPath = []
        self.DFS(src, dst, SDpair)
        return self.pathList

    def pathWidth(self, path, SDpair):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i - 1]
            width = min(width, self.f_i_LP[SDpair][(currentNode, nextNode)])

        return width

    def DFS(self, currentNode: Node, dst: Node, SDpair):
        self.currentPath.append(currentNode)
        if currentNode == dst:
            self.pathList.append(self.currentPath[::-1])
        else:
            adjcentNode = set()
            for link in currentNode.links:
                nextNode = link.theOtherEndOf(currentNode)
                if self.f_i_LP[SDpair][(currentNode, nextNode)] >= 1:
                    adjcentNode.add(nextNode)

            for node in adjcentNode:
                if node not in self.currentPath:
                    self.DFS(node, dst, SDpair)

        self.currentPath.pop()

if __name__ == '__main__':
    
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99]), (topo.nodes[0], topo.nodes[98])], 1)