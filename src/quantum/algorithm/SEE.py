import sys
import math
import random
import gurobipy as gp
from gurobipy import quicksum
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from numpy import log as ln
from random import sample


EPS = 1e-6
class SEE(AlgorithmBase):
    def __init__(self, topo):
        super().__init__(topo)
        self.name = "REPS"
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0

    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        return (varName + str(parName)).replace(' ', '').replace(',', '][')
    
    def printResult(self):
        self.topo.clearAllEntanglements()
        self.result.waitingTime = self.totalWaitingTime / self.totalRequest
        self.result.usedQubits = self.totalUsedQubits / self.totalRequest
        
        # self.result.remainRequestPerRound.append(len(self.requests) / self.totalRequest)
        self.result.remainRequestPerRound.append(len(self.requests))
        
        print("[REPS] total time:", self.result.waitingTime)
        print("[REPS] remain request:", len(self.requests))
        print("[REPS] current Timeslot:", self.timeSlot)



        print('[REPS] idle time:', self.result.idleTime)
        print('[REPS] remainRequestPerRound:', self.result.remainRequestPerRound)
        print('[REPS] avg usedQubits:', self.result.usedQubits)

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))

        self.srcDstPairs = []
        for request in self.requests:
            src = request[0]
            dst = request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime += len(self.requests)
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.EPI()
            self.ELS()

        # print('[REPS] p2 end')
    
    def p4(self):
        # if len(self.srcDstPairs) > 0:
            # self.EPI()
        # print('[REPS] p4 end') 
        self.printResult()
        return self.result




    
    
    

    
   
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
    def edgeFullCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        return capacity
    def segmentCapacity(self, u, v):
        min_capacity = 1000
        for path , l in self.topo.k_shortest_paths(u , v , 5):
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i+1]
                capacity = 0
                for link in self.topo.nodes[p1].links:
                    if link.contains(self.topo.nodes[p2]):
                        capacity += 1
                if capacity < min_capacity:
                    min_capacity = capacity
        return min_capacity


    def widthForSort(self, path):
        # path[-1] is the path of weight
        return -path[-1]
    
    
            
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled:
                capacity += 1
        return capacity

    def entanglementProb(self, u , v , k):
        paths_with_len = self.topo.k_shortest_paths(u , v , 5)
        path , len = paths_with_len[k]
        # print('math.exp(-self.topo.alpha * len) ' , len , math.exp(-self.topo.alpha * len))

        return math.exp(-self.topo.alpha * len)

    def LP1(self):
        # print('[REPS] LP1 start')
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        # numOfFlow = [self.ti[self.srcDstPairs[i]] for i in range(numOfSDpairs)]
        numOfFlow = [2 for i in range(numOfSDpairs)]
        if len(numOfFlow):
            maxN = max(numOfFlow)
        else:
            maxN = 0
        # maxK = 2
        print('maxN ' , maxN)
        self.fki_LP = {SDpair : [{} for _ in range(maxN)] for SDpair in self.srcDstPairs}
        self.tki_LP = {SDpair : [0] * maxN for SDpair in self.srcDstPairs}
        
        edgeIndices = []
        notEdge = []
        self.segments = []
        K_u_v = {}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                K_u_v[(node1 , node2)] = 0
                


        for edge in self.topo.edges:
            edgeIndices.append((edge[0].id, edge[1].id))
            self.segments.append(edge)

        
        # for u in range(numOfNodes):
        #     for v in range(numOfNodes):
        #         if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
        #             notEdge.append((u, v))

        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if node1 == node2:
                    notEdge.append((node1.id , node2.id))
                elif  self.topo.distance_by_node(node1.id , node2.id) <=500 and ((node1.id , node2.id) not in edgeIndices) and ((node2.id , node1.id) not in edgeIndices):
                    edgeIndices.append((node1.id , node2.id))
                    self.segments.append((node1 , node2))
                else:
                    notEdge.append((node1.id , node2.id))

        print('len(edgeIndices)' , len(edgeIndices))
        print('len(self.topo.edges)' , len(self.topo.edges))
        print('len(self.segments)' , len(self.segments))



        m = gp.Model('SEE for EPI')
        m.setParam("OutputFlag", 0)

        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxN
            for n in range(maxN):
                f[i][n] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][n][u] = [0] * numOfNodes 
                    for v in range(numOfNodes):
                        if n < numOfFlow[i] and ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            f[i][n][u][v] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, n, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxN
            for n in range(maxN):
                if n < numOfFlow[i]:
                    t[i][n] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, n))
                else:
                    t[i][n] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, n))
        x = [0] * numOfNodes
        for u in range(numOfNodes):
            x[u] = [0] * numOfNodes
            for v in range(numOfNodes):
                # if ((u, v) in edgeIndices or (v, u) in edgeIndices):
                    x[u][v] = [0] * len(self.topo.k_shortest_paths(u , v , 5))
                    for k in range(len(self.topo.k_shortest_paths(u , v , 5))): #later we'll find this
                        if ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            x[u][v][k] = m.addVar(lb = 0 , vtype = gp.GRB.CONTINUOUS , name = "x[%d][%d][%d]"%(u,v,k))

        m.update()
        
        m.setObjective(quicksum(t[i][n] for n in range(maxN) for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            d = self.srcDstPairs[i][1].id
            
            for n in range(numOfFlow[i]):
                neighborOfS = []
                neighborOfD = []

                for edge in edgeIndices:
                    if edge[0] == s:
                        neighborOfS.append(edge[1])
                    elif edge[1] == s:
                        neighborOfS.append(edge[0])
                    if edge[0] == d:
                        neighborOfD.append(edge[1])
                    elif edge[1] == d:
                        neighborOfD.append(edge[0])
                    
                m.addConstr(quicksum(f[i][n][s][v] for v in neighborOfS) - quicksum(f[i][n][v][s] for v in neighborOfS) == t[i][n])
                m.addConstr(quicksum(f[i][n][d][v] for v in neighborOfD) - quicksum(f[i][n][v][d] for v in neighborOfD) == -t[i][n])

                # for u in range(numOfNodes):
                #     if u not in [s, d]:
                #         edgeUV = []
                #         for v in range(numOfNodes):
                #             if v not in [s, d]:
                #                 edgeUV.append(v)
                        # m.addConstr(quicksum(f[i][n][u][v] for v in edgeUV) - quicksum(f[i][n][v][u] for v in edgeUV) == 0)
                # for (u,v) in edgeIndices:
                #     if u not in [s,d] and v not in [s,d]:
                m.addConstr(quicksum(f[i][n][u][v] for (u,v) in edgeIndices) - quicksum(f[i][n][v][u] for (u,v) in edgeIndices) == 0)


        
        # for (u, v) in edgeIndices:
        #     m.addConstr(quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= quicksum(self.entanglementProb(u , v , k) * x[u][v][k]* math.sqrt(self.topo.nodes[u].q*self.topo.nodes[v].q) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))
        # for u in range(numOfNodes):
        #     for v in range(numOfNodes):
        #         # if ((u, v) in edgeIndices or (v, u) in edgeIndices):
        #         # m.addConstr(quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= quicksum((self.entanglementProb(u , v , k) * x[u][v][k]* math.sqrt(self.topo.nodes[u].q*self.topo.nodes[v].q)) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))
        #         m.addConstr(quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= quicksum((self.entanglementProb(u , v , k) * x[u][v][k]) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))

        for (u,v) in edgeIndices:
                m.addConstr(quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= quicksum((self.entanglementProb(u , v , k) * x[u][v][k]* math.sqrt(self.topo.nodes[u].q*self.topo.nodes[v].q)) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))
                # m.addConstr(quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= quicksum((self.entanglementProb(u , v , k) * x[u][v][k]) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))


        test_dict = {(u ,v , 0):0 for (u,v) in edgeIndices}
        for edge in self.topo.edges:
            e1 = edge[0].id
            e2 = edge[1].id
            segContainingEdge = set()
            segContainingEdgeList = [] 
            for (u,v) in edgeIndices:
                paths_with_len = self.topo.k_shortest_paths(u , v , 5)
                j=0
                for path, l in paths_with_len:
                    for i in range(len(path) - 1):
                        p1 = path[i]
                        p2 = path[i+1]
                        if (e1,e2) == (p1 , p2) or (e1,e2) == (p2,p1):
                            # print('in 2nd constr ' , (e1,e2), (p1 , p2) )
                            segContainingEdge.add((u,v , j))
                            segContainingEdgeList.append((u,v , j))
                           
                        
                    j+=1
            capacity = self.edgeFullCapacity(edge[0] , edge[1])
            m.addConstr(quicksum(x[n1][n2][k] for (n1,n2 ,k) in segContainingEdge) <= capacity)


        # for (u,v) in edgeIndices:
        #     capacity = self.segmentCapacity(u,v)
        #     m.addConstr(quicksum(x[u][v][k] for k in range(len(self.topo.k_shortest_paths(u , v , 5))) ) <= capacity)



        for (u,v) in edgeIndices:
            mu = min(self.topo.nodes[u].remainingQubits , self.topo.nodes[v].remainingQubits)
            # print('memory for ' , (u,v) , mu)
            m.addConstr(quicksum(x[u][v][k] for k in range(len(self.topo.k_shortest_paths(u , v , 5))) ) <= min(self.topo.nodes[u].remainingQubits , self.topo.nodes[v].remainingQubits))
            # m.addConstr(quicksum(x[u][v][k] for k in range(len(self.topo.k_shortest_paths(u , v , 5))) ) <= self.topo.nodes[v].remainingQubits)

        # for u in range(numOfNodes):
        #     segContainu = []
        #     for (n1, n2) in edgeIndices:
        #         if u == n1:
        #             segContainu.append((n1, n2))
        #         elif u == n2:
        #             segContainu.append((n2, n1))
        #     # print('len(edgeContainu)' , len(edgeContainu))
        #     # print('self.topo.nodes[u].remainingQubits' , self.topo.nodes[u].remainingQubits)
        #     m.addConstr(quicksum(x[n1][n2][k] for (n1, n2) in segContainu for k in range(len(self.topo.k_shortest_paths(n1 , n2 , 5)))) <= self.topo.nodes[u].remainingQubits)

        # for i in range(numOfSDpairs):
        #     for n in range(numOfFlow[i] - 1):
        #         m.addConstr(t[i][n] >= t[i][n+1])


        print('optimize start....')
        m.optimize()

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]

            for n in range(numOfFlow[i]):
                for edge in self.segments:
                    u = edge[0]
                    v = edge[1]
                    varName = self.genNameByBbracket('f', [i, n, u.id, v.id])
                    self.fki_LP[SDpair][n][(u, v)] = m.getVarByName(varName).x
                    # if self.fki_LP[SDpair][n][(u, v)] > 0:
                    #     print('# ',SDpair[0].id ,SDpair[1].id, '  ' , u.id , v.id , '  '  , self.fki_LP[SDpair][n][(u, v)])

                for edge in self.segments:
                    u = edge[1]
                    v = edge[0]
                    varName = self.genNameByBbracket('f', [i, n, u.id, v.id])
                    self.fki_LP[SDpair][n][(u, v)] = m.getVarByName(varName).x
                    # if self.fki_LP[SDpair][n][(u, v)] > 0:
                    #     print('# ',SDpair[0].id ,SDpair[1].id, '  ' , u.id , v.id , '  '  , self.fki_LP[SDpair][n][(u, v)])
                

                # for (u, v) in notEdge:
                #     u = self.topo.nodes[u]
                #     v = self.topo.nodes[v]
                #     self.fki_LP[SDpair][n][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, n])
                self.tki_LP[SDpair][n] = m.getVarByName(varName).x
                print('* ' , self.tki_LP[SDpair][n])
        print('[SEE] LP1 end')

    def EPI(self):
        self.LP1()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : 2 for SDpair in self.srcDstPairs}
        self.fki = {SDpair : [{} for n in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.tki = {SDpair : [0 for n in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.pathForELS = {SDpair : [] for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for n in range(numOfFlow[SDpair]):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][n][(u, v)] = 0

        for SDpair in self.srcDstPairs:
            for n in range(numOfFlow[SDpair]):
                self.tki[SDpair][n] = self.tki_LP[SDpair][n] >= random.random()
                # print('for SD ' , SDpair[0].id , SDpair[1].id , self.tki_LP[SDpair][n])

                if not self.tki[SDpair][n]:
                    continue
                paths = self.findPathsForEPS(SDpair, n)

                print('for SD ' , SDpair[0].id , SDpair[1].id  , len(paths))

                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][n][(u, v)] = 0
                    
                for path in paths:

                    width = path[-1]
                    select = (width / self.tki_LP[SDpair][n]) >= random.random()
                    print('path ', select , [p.id for p in path[0:-1]] , width)

                    if not select:
                        continue
                    path = path[:-1]
                    self.pathForELS[SDpair].append(path)

                    pathLen = len(path)
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        self.fki[SDpair][n][(node, next)] = 1
            print('path for els ' , len(self.pathForELS[SDpair]))
                
        print('[REPS] EPI end')

    def ESC(self):
        T = self.pathForELS
        self.x = {(u, v , k) : 0 for u in self.topo.nodes for v in self.topo.nodes for k in range(len(self.topo.k_shortest_paths(u , v , 5)))}
        self.D = []
        for SDPair in T:
            for path in T[SDPair]:
                self.D.append(path)
                for i in range(len(path) -1):
                    n1 = path[i]
                    n2 = path[i+1]
                    



    def ELS(self):
        Ci = self.pathForELS
        self.y = {(u, v) : 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        needLink = {}
        nextLink = {node : [] for node in self.topo.nodes}
        Pi = {SDpair : [] for SDpair in self.srcDstPairs}
        T = [SDpair for SDpair in self.srcDstPairs]

        while len(T) > 0:
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangle(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths:
                    Ci[SDpair].remove(path)
                if len(Ci[SDpair]) == 0 and SDpair in T:
                    T.remove(SDpair)
            
            if len(T) == 0:
                break

            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) < minLength:
                        minLength = len(path)
                        i = SDpair
            
            src = i[0]
            dst = i[1]

            minR = math.inf
            for path in Ci[i]:
                r = 0
                for node in path:
                    r += self.weightOfNode[node]
                if minR > r:
                    targetPath = path
                    minR = r
            
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []

            Pi[i].append(targetPath)
            for nodeIndex in range(1, len(targetPath) - 2):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                for link in node.links:
                    if link.contains(next) and link.entangled and link.notSwapped():
                        targetLink1 = link
                    
                    if link.contains(prev) and link.entangled and link.notSwapped():
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1

                nextLink[node].append(targetLink1)
                needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))

            T.remove(i)

        T = [SDpair for SDpair in self.srcDstPairs]
        while len(T) > 0:
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangle(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths:
                    Ci[SDpair].remove(path)
            
            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) - 1 < minLength:
                        minLength = len(path) - 1
                        i = SDpair
                if len(Ci[SDpair]) == 0 and i == -1:
                    i = SDpair
            
            src = i[0]
            dst = i[1]

            targetPath = self.findPathForELS(i)
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []
            Pi[i].append(targetPath)
            for nodeIndex in range(1, len(targetPath) - 1):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                for link in node.links:
                    if link.contains(next) and link.entangled:
                        targetLink1 = link
                    
                    if link.contains(prev) and link.entangled:
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1
                nextLink[node].append(targetLink1)
                needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))
            T.remove(i)
        
        # print('[REPS] ELS end')
        # print('[REPS]' + [(src.id, dst.id) for (src, dst) in self.srcDstPairs])
        for SDpair in self.srcDstPairs:
            src = SDpair[0]
            dst = SDpair[1]

            if len(Pi[SDpair]):
                self.result.idleTime -= 1

            for pathIndex in range(len(Pi[SDpair])):
                path = Pi[SDpair][pathIndex]
                # print('[REPS] attempt:', [node.id for node in path])
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    node.attemptSwapping(link1, link2)
                successPath = self.topo.getEstablishedEntanglements(src, dst)
                # for x in successPath:
                #     print('[REPS] success:', [z.id for z in x])

                if len(successPath):
                    for request in self.requests:
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(request)
                            break
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    link1.clearPhase4Swap()
                    link2.clearPhase4Swap()


    
   
        

   
    
    def findPathsForEPS(self, SDpair, n):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForEPS(SDpair, n):
            print('found dijkstra =====================')
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForEPS(path, SDpair, n)
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fki_LP[SDpair][n][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForEPS(self, SDpair, n):
        print('-----------')
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        # for node1 in self.topo.nodes:
        #     for node2 in self.topo.nodes:
        #         if self.edgeSuccessfulEntangle(node1, node2) > 0:
        #             adjcentList[node1].add(node2)
        for node1 , node2 in self.segments:
            adjcentList[node1].add(node2)
            adjcentList[node2].add(node1)
        # print('adjacent list for ' , src.id , [x.id for x in adjcentList[src]])
        
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
            # print('in while loop adjacent list for ' , u.id , [x.id for x in adjcentList[u]])
            
            for next in adjcentList[u]:
                # print('inside loop1 ' , distance[u] , self.fki_LP[SDpair][n][(u, next)] )
                newDistance = min(distance[u], self.fki_LP[SDpair][n][(u, next)])
                # print('inside loop1 ' , distance[next] , newDistance )

                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    # print('parent of ' , next.id , u.id)
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
    
    def findPathForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        if self.DijkstraForELS(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]
            path = path[::-1]
            return path
        else:
            return []
    
    def DijkstraForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.y[(node1, node2)] < self.edgeSuccessfulEntangle(node1, node2):
                    adjcentList[node1].add(node2)
        
        distance = {node : math.inf for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((self.weightOfNode[src], src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = distance[u] + self.weightOfNode[next]
                if distance[next] > newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((distance[next], next.id))

        return False
if __name__ == '__main__':
    
    topo = Topo.generate(10, 0.9, 5, 0.0002, 6)
    s = SEE(topo)
    result = AlgorithmResult()
    samplesPerTime = 2
    ttime = 1
    rtime = 1
    requests = {i : [] for i in range(ttime)}
    Ni = 5

    for i in range(ttime):
        if i < rtime:

            # ids = [(96, 71), (99, 40)]
            # for (p,q) in ids:
            #     source = None
            #     dest = None
            #     for node in topo.nodes:

            #         if node.id == p:
            #             source = node
            #         if node.id == q:
            #             dest = node
            #     requests[i].append((source , dest))

            a = sample(topo.nodes, samplesPerTime)
            for n in range(0,samplesPerTime,2):
                requests[i].append((a[n], a[n+1]))
        print('[REPS] S/D:' , i , [(a[0].id , a[1].id) for a in requests[i]])

    for i in range(ttime):
        result = s.work(requests[i], i)
    

    # print(result.waitingTime, result.numOfTimeslot)