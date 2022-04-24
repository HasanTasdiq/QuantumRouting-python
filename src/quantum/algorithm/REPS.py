import sys
import math
import random
import gurobipy as gp
from gurobipy import quicksum
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link

EPS = 1e-6
class REPS(AlgorithmBase):
    def __init__(self, topo):
        super().__init__(topo)
        self.name = "REPS"
        self.F = open("output.txt", "w")

    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        if len(parName) == 0:
            return varName
        name = varName
        name += "["
        for par in parName:
            name += (str(par) + "][")
        name = name[0: len(name) - 2] + "]"
        return name
    def p2(self):
        self.PFT() # compute (self.ti, self.fi)
        for SDpair in self.srcDstPairs:
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                need = self.fi[SDpair][(u, v)] + self.fi[SDpair][(u, v)]
                if self.fi[SDpair][edge]:
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
    
    # return fi(u, v)

    def LP1(self):
        # initialize fi(u, v) ans ti

        self.fi_LP = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti_LP = {SDpair : 0 for SDpair in self.srcDstPairs}
        
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

        m = gp.Model('REPS for PFT')
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

            capacity = self.edgeCapacity(self.topo.nodes[u], self.topo.nodes[v])
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
            SDpair = self.srcDstPairs[i]
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                varName = self.genNameByComma('f', [i, u.id, v.id])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for edge in self.topo.edges:
                u = edge[1]
                v = edge[0]
                varName = self.genNameByComma('f', [i, u.id, v.id])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for (u, v) in notEdge:
                u = self.topo.nodes[u]
                v = self.topo.nodes[v]
                self.fi_LP[SDpair][(u, v)] = 0
            
            
            varName = self.genNameByComma('t', [i])
            self.ti_LP[SDpair] = m.getVarByName(varName).x
        
    def edgeCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        used = 0
        for SDpair in self.srcDstPairs:
            used += self.fi[SDpair][(u, v)]
            used += self.fi[SDpair][(v, u)]
        return capacity - used

    def widthForSort(self, path):
        # path[-1] is the path of weight
        return -path[-1]
    
    def PFT(self):

        # initialize fi and ti
        self.fi = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti = {SDpair : 0 for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for u in self.topo.nodes:
                for v in self.topo.nodes:
                    self.fi[SDpair][(u, v)] = 0
        
        # PFT
        failedFindPath = False
        while not failedFindPath:
            self.LP1()
            failedFindPath = True
            Pi = {}
            paths = []
            for SDpair in self.srcDstPairs:
                Pi[SDpair] = self.findPathsForPFT(SDpair)

            for SDpair in self.srcDstPairs:
                K = len(Pi[SDpair])
                for k in range(K):
                    width = math.floor(Pi[SDpair][k][-1])
                    Pi[SDpair][k][-1] -= width
                    paths.append(Pi[SDpair][k])
                    pathLen = len(Pi[SDpair][k]) - 1
                    self.ti[SDpair] += width
                    if width > 0:
                        failedFindPath = False

                    for i in range(pathLen - 1):
                        node = Pi[SDpair][k][i]
                        next = Pi[SDpair][k][i + 1]
                        self.fi[SDpair][(node, next)] += width

            sorted(paths, key=self.widthForSort)

            for path in paths:
                pathLen = len(path) - 1
                width = path[-1]
                SDpair = (path[0], path[-2])
                isable = True
                for i in range(pathLen - 1):
                    node = path[i]
                    next = path[i + 1]
                    if self.edgeCapacity(node, next) >= 1:
                        isable = False
                
                if not isable:
                    for i in range(pathLen - 1):
                        node = path[i]
                        next = path[i + 1]
                        self.fi_LP[SDpair][(node, next)] -= width
                    continue
                    
                failedFindPath = False
                self.ti[SDpair] += 1
                for i in range(pathLen - 1):
                    node = path[i]
                    next = path[i + 1]
                    self.fi[SDpair][(node, next)] += 1

        print('PFT end')

            
    def edgeSuccessfulEntangle(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled:
                capacity += 1
        return capacity

    def LP2(self):
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow = [self.ti[self.srcDstPairs[i]] for i in range(numOfSDpairs)]
        maxK = max(numOfFlow)
        self.fki_LP = {SDpair : [{} for _ in range(maxK)] for SDpair in self.srcDstPairs}
        self.tki_LP = {SDpair : [0] * maxK for SDpair in self.srcDstPairs}
        
        edgeIndices = []
        notEdge = []
        for edge in self.topo.edges:
            edgeIndices.append((edge[0].id, edge[1].id))
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))

        m = gp.Model('REPS for EPS')

        
        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxK
            for k in range(maxK):
                f[i][k] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][k][u] = [0] * numOfNodes 
                    for v in range(numOfNodes):
                        if k < numOfFlow[i]:
                            f[i][k][u][v] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, k, u, v))
                        else:
                            f[i][k][u][v] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, k, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxK
            for k in range(maxK):
                if k < numOfFlow[i]:
                    t[i][k] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))
                else:
                    t[i][k] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))

        m.update()
        
        m.setObjective(quicksum(t[i][k] for k in range(maxK) for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            d = self.srcDstPairs[i][1].id
            
            for k in range(numOfFlow[i]):
                m.addConstr(quicksum(f[i][k][s][v] for v in range(numOfNodes)) - quicksum(f[i][k][v][s] for v in range(numOfNodes)) == t[i][k])
                m.addConstr(quicksum(f[i][k][d][v] for v in range(numOfNodes)) - quicksum(f[i][k][v][d] for v in range(numOfNodes)) == -t[i][k])

                for u in range(numOfNodes):
                    if u not in [s, d]:
                        m.addConstr(quicksum(f[i][k][u][v] for v in range(numOfNodes)) - quicksum(f[i][k][v][u] for v in range(numOfNodes)) == 0)

        
        for (u, v) in edgeIndices:
            capacity = self.edgeSuccessfulEntangle(self.topo.nodes[u], self.topo.nodes[v])
            m.addConstr(quicksum(f[i][k][u][v] + f[i][k][v][u] for k in range(numOfFlow[i]) for i in range(numOfSDpairs)) <= capacity)

        for (u, v) in notEdge:
            for i in range(numOfSDpairs):
                for k in range(numOfFlow[i]):
                    m.addConstr(f[i][k][u][v] == 0)

        m.optimize()

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]

            for k in range(numOfFlow[i]):
                for edge in self.topo.edges:
                    u = edge[0]
                    v = edge[1]
                    varName = self.genNameByBbracket('f', [i, k, u.id, v.id])
                    self.fki_LP[SDpair][k][(u, v)] = m.getVarByName(varName).x

                for edge in self.topo.edges:
                    u = edge[1]
                    v = edge[0]
                    varName = self.genNameByBbracket('f', [i, k, u.id, v.id])
                    self.fki_LP[SDpair][k][(u, v)] = m.getVarByName(varName).x

                for (u, v) in notEdge:
                    u = self.topo.nodes[u]
                    v = self.topo.nodes[v]
                    self.fki_LP[SDpair][k][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, k])
                print(k)
                self.tki_LP[SDpair][k] = m.getVarByName(varName).x
    
    def EPS(self):
        self.LP2()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : self.ti[SDpair] for SDpair in self.srcDstPairs}
        self.fki = {SDpair : [{} for _ in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.tki = {SDpair : [self.tki_LP[SDpair][k] >= random.random() for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                if not self.tki[SDpair][k]:
                    continue

        print('EPS end')

    def ELS(self):
        print('ELS end')


    def findPathsForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForPFT(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForPFT(path, SDpair)
            pathLen = len(path)
            for i in range(pathLen - 1):
                node = path[i]
                next = path[i + 1]
                self.fi_LP[SDpair][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node in self.topo.nodes:
            for link in node.links:
                neighbor = link.theOtherEndOf(node)
                adjcentList[node].add(neighbor)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fi_LP[SDpair][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], next.id))

        return False

    def widthForPFT(self, path, SDpair):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fi_LP[SDpair][(currentNode, nextNode)])

        return width
    
    def findPathsForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForEPS(SDpair, k):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForEPS(path, SDpair)
            pathLen = len(path)
            for i in range(pathLen - 1):
                node = path[i]
                next = path[i + 1]
                self.fki_LP[SDpair][k][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node in self.topo.nodes:
            for link in node.links:
                neighbor = link.theOtherEndOf(node)
                adjcentList[node].add(neighbor)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fki_LP[SDpair][k][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], next.id))

        return False

    def widthForEPS(self, path, SDpair, k):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fki_LP[SDpair][k][(currentNode, nextNode)])

        return width

if __name__ == '__main__':
    
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = REPS(topo)
    s.work([(topo.nodes[3],topo.nodes[99]), (topo.nodes[0], topo.nodes[98])], 1)