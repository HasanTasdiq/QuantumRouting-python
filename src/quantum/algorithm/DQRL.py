import sys
import math
import random
import gurobipy as gp
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from topo.helper import request_timeout
from numpy import log as ln
from random import sample
import numpy as np

sys.path.insert(0, "../../rl")

from DQRLAgent import DQRLAgent

EPS = 1e-6
class QuRA_DQRL(AlgorithmBase):
    def __init__(self, topo,param=None, name=''):
        super().__init__(topo)
        self.name = name
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        # self.entAgent = DQNAgentDistEnt(self, 0)
        self.routingAgent = DQRLAgent(self , 0)

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
        print('[' , self.name, '] :' , self.timeSlot, ' total successful request::', self.result.successfulRequest)

        print('[REPS] remainRequestPerRound:', self.result.remainRequestPerRound[-1])
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
            # self.PFT() # compute (self.ti, self.fi)
            # self.randPFT()
            self.entAgent.learn_and_predict()
        # print('[REPS] p2 end')
    
    def p4(self):

        if len(self.srcDstPairs) > 0:
            # self.EPS()
            # self.ELS()
            self.route()
        # print('[REPS] p4 end') 
        self.printResult()
        self.entAgent.update_reward()
        self.routingAgent.update_reward()
        return self.result
    def route(self):
        successReq = 0
        totalEntanglement = 0
        for request in self.srcDstPairs:
            
            print((request[0].id, request[1].id))

            src,dst = request[0] , request[1]
            current_node = request[0]
            prev_node = None
            prev_links = []
            hopCount = 0
            success = False
            good_to_search = True
            width = 0
            usedLinks = []
            selectedNodes = []
            selectedEdges = []
            while (not current_node == request[1]) and (hopCount < 15) and good_to_search:
                current_state, next_node_id = self.routingAgent.learn_and_predict_next_node(request, current_node)
                next_node = self.topo.nodes[next_node_id]
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
                print('========ent_links=======')
                print(current_node.id , next_node_id , len(ent_links))
                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node_id)

                if not len(ent_links):
                    good_to_search = False

                if current_node == next_node:
                    good_to_search = False
                    
                if next_node in selectedNodes:
                    good_to_search = False
                
                if good_to_search:
                    try:
                        self.topo.reward_swap[key] += self.topo.positive_reward
                    except:
                        self.topo.reward_swap[key] = self.topo.positive_reward
                    
                if prev_node is not None:
                    if good_to_search:
                        swapCount = 0
                        swappedlinks = []
                        for link1,link2 in zip(prev_links , ent_links):
                            swapped = current_node.attemptSwapping(link1, link2)
                            if swapped:
                                swappedlinks.append(link2)
                            usedLinks.append(link2)
                        if len(swappedlinks):
                            prev_links = swappedlinks
                        else:
                            good_to_search = False
                            
                else:
                    prev_links = ent_links
                    usedLinks.extend(prev_links)
                
                self.routingAgent.update_action( request ,  next_node_id  , current_state )
                
                if len(prev_links) and next_node == request[1]:
                    success = True
                    break
                else:
                    selectedNodes.append(next_node)
                    selectedEdges.append((current_node, next_node))
                    
                prev_node = current_node
                current_node = next_node
                hopCount += 1
            
            if success:
                successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)

                for path in successPath:
                    for node, link in path:
                        if link is not None:
                            link.used = True
                            edge = self.topo.linktoEdgeSorted(link)

                            self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward
                        # if node is not None:
                        #     try:
                        #         self.topo.reward_swap[(request , node)] += self.topo.positive_reward
                        #     except:
                        #         self.topo.reward_swap[(request , node)] = self.topo.positive_reward
                        #                     self.topo.reward_ent[edge] = self.topo.negative_reward

                for (current_node, next_node) in selectedEdges:
                    key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node_id)

                    try:
                        self.topo.reward_swap[key] += self.topo.positive_reward*2
                    except:
                        self.topo.reward_swap[key] = self.topo.positive_reward*2
                print("!!!!!!!success!!!!!!!")

                # print([(r[0].id, r[1].id) for r in self.requests])

                # self.requests.remove(request)
                for req in self.requests:
                    src = req[0]
                    dst = req[1]
                    if (src, dst) == (request[0], request[1]):
                        # print('[REPS] finish time:', self.timeSlot - request[2])
                        self.requests.remove(req)

                successReq += 1
                totalEntanglement += len(successPath)
            else:
                for link in usedLinks:
                        edge = self.topo.linktoEdgeSorted(link)
                        try:
                            self.topo.reward_ent[edge] += self.topo.negative_reward
                        except:
                            self.topo.reward_ent[edge] = self.topo.negative_reward
                for (current_node, next_node) in selectedEdges:
                    key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node_id)

                    try:
                        self.topo.reward_swap[key] += self.topo.negative_reward
                    except:
                        self.topo.reward_swap[key] = self.topo.negative_reward
            
            for link in usedLinks:
                link.clearPhase4Swap()


        self.result.usedLinks += len(usedLinks)
        
        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request:', successReq)
                    


    
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
    
        
          
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled:
                capacity += 1
        return capacity

    def LP2(self):
        # print('[REPS] LP2 start')
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow = [9 for i in range(numOfSDpairs)]
        if len(numOfFlow):
            maxK = max(numOfFlow)
        else:
            maxK = 0
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
        m.setParam("OutputFlag", 0)

        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxK
            for k in range(maxK):
                f[i][k] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][k][u] = [0] * numOfNodes 
                    for v in range(numOfNodes):
                        if k < numOfFlow[i] and ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            f[i][k][u][v] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, k, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxK
            for k in range(maxK):
                if k < numOfFlow[i]:
                    t[i][k] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))
                else:
                    t[i][k] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))

        m.update()
        
        m.setObjective(gp.quicksum(t[i][k] for k in range(maxK) for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            d = self.srcDstPairs[i][1].id
            
            for k in range(numOfFlow[i]):
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
                    
                m.addConstr(gp.quicksum(f[i][k][s][v] for v in neighborOfS) - gp.quicksum(f[i][k][v][s] for v in neighborOfS) == t[i][k])
                m.addConstr(gp.quicksum(f[i][k][d][v] for v in neighborOfD) - gp.quicksum(f[i][k][v][d] for v in neighborOfD) == -t[i][k])

                for u in range(numOfNodes):
                    if u not in [s, d]:
                        edgeUV = []
                        for v in range(numOfNodes):
                            if v not in [s, d]:
                                edgeUV.append(v)
                        m.addConstr(gp.quicksum(f[i][k][u][v] for v in edgeUV) - gp.quicksum(f[i][k][v][u] for v in edgeUV) == 0)

        
        for (u, v) in edgeIndices:
            capacity = self.edgeSuccessfulEntangle(self.topo.nodes[u], self.topo.nodes[v])
            m.addConstr(gp.quicksum((f[i][k][u][v] + f[i][k][v][u]) for k in range(maxK) for i in range(numOfSDpairs)) <= capacity)

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

                # for (u, v) in notEdge:
                #     u = self.topo.nodes[u]
                #     v = self.topo.nodes[v]
                #     self.fki_LP[SDpair][k][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, k])
                self.tki_LP[SDpair][k] = m.getVarByName(varName).x
        # print('[REPS] LP2 end')

    def EPS(self):
        self.LP2()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : 9 for SDpair in self.srcDstPairs}

        self.fki = {SDpair : [{} for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.tki = {SDpair : [0 for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.pathForELS = {SDpair : [] for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                self.tki[SDpair][k] = self.tki_LP[SDpair][k] >= random.random()

                if not self.tki[SDpair][k]:
                    continue
                paths = self.findPathsForEPS(SDpair, k)

                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0
                    
                for path in paths:
                    width = path[-1]
                    select = (width / self.tki_LP[SDpair][k]) >= random.random()
                    # print('*** path ', select , [p.id for p in path[0:-1]] , width)

                    if not select:
                        continue
                    path = path[:-1]
                    self.pathForELS[SDpair].append(path)
                    pathLen = len(path)
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        self.fki[SDpair][k][(node, next)] = 1
                
        print('[REPS] EPS end')

    def ELS(self):
        Ci = self.pathForELS
        self.y = {(u, v) : 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        needLink = {}
        nextLink = {node : [] for node in self.topo.nodes}
        Pi = {SDpair : [] for SDpair in self.srcDstPairs}
        T = [SDpair for SDpair in self.srcDstPairs]
        output = []
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
            output.append(targetPath)
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
        print('** before graph ' , len(output))
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
        totalEntanglement = 0
        successReq = 0
        usedLinks = set()

        for SDpair in self.srcDstPairs:
            src = SDpair[0]
            dst = SDpair[1]

            if len(Pi[SDpair]):
                self.result.idleTime -= 1

            for pathIndex in range(len(Pi[SDpair])):
                path = Pi[SDpair][pathIndex]
                # print('[REPS] attempt:', [node.id for node in path])
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    usedLinks.add(link1)
                    usedLinks.add(link2)
                    node.attemptSwapping(link1, link2)
                successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)

                for path in successPath:
                    for node, link in path:
                        if link is not None:
                            link.used = True
                            edge = self.topo.linktoEdgeSorted(link)

                            self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward

                            # self.result.usedLinks += 1
                # for x in successPath:
                #     print('[REPS] success:', [z.id for z in x])

                if len(successPath):
                    for request in self.requests:
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(request)
                            successReq += 1
                            break
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    if not link1 is None and not link1.used and link1.entangled:
                        edge = self.topo.linktoEdgeSorted(link1)
                        
                        try:
                            self.topo.reward_ent[edge] += self.topo.negative_reward
                        except:
                            self.topo.reward_ent[edge] = self.topo.negative_reward

                    if not link2 is None and not link2.used and link2.entangled:

                        edge = self.topo.linktoEdgeSorted(link2)
                        
                        try:
                            self.topo.reward_ent[edge] += self.topo.negative_reward
                        except:
                            self.topo.reward_ent[edge] = self.topo.negative_reward
                    link1.clearPhase4Swap()
                    link2.clearPhase4Swap()
                
                totalEntanglement += len(successPath)
        self.result.usedLinks += len(usedLinks)
        
        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request:', successReq)

    def filterReqeuest(self):
        self.requests = list(filter(lambda x: self.timeSlot -  x[2] < self.topo.requestTimeout -1 , self.requests))

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
            
            for i in range(len(path) - 1):
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
            width = self.widthForEPS(path, SDpair, k)
            for i in range(len(path) - 1):
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
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.edgeSuccessfulEntangle(node1, node2) > 0:
                    adjcentList[node1].add(node2)
        
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
    
    topo = Topo.generate(50, 0.9, 5, 0.0002, 6)
    s = DQRL(topo)
    result = AlgorithmResult()
    samplesPerTime = 8 * 2
    ttime = 100
    rtime = ttime
    # requests = {i : [] for i in range(ttime)}

    # for i in range(ttime):
    #     if i < rtime:

    #         # ids =  [(1,15), (1,16), (4,17), (3,16)]

    #         # for (p,q) in ids:
    #         #     source = None
    #         #     dest = None
    #         #     for node in topo.nodes:

    #         #         if node.id == p:
    #         #             source = node
    #         #         if node.id == q:
    #         #             dest = node
    #         #     requests[i].append((source , dest))

    #         a = sample(topo.nodes, samplesPerTime)
    #         for n in range(0,samplesPerTime,2):
    #             requests[i].append((a[n], a[n+1]))
    #     print('[REPS] S/D:' , i , [(a[0].id , a[1].id) for a in requests[i]])

    # for i in range(ttime):
    #     result = s.work(requests[i], i)
    

    for i in range(0, 100):
        requests = []
        if i < 100:
            for j in range(20):
                a = sample(topo.nodes, 2)
                requests.append((a[0], a[1]))
            
            # ids = [(1,15), (1,16), (4,17), (3,16)]
            # for (p,q) in ids:
            #     source = None
            #     dest = None
            #     for node in topo.nodes:

            #         if node.id == p:
            #             source = node
            #         if node.id == q:
            #             dest = node
            #     requests.append((source , dest))

            s.work(requests, i)
        else:
            s.work([], i)

    # print(result.waitingTime, result.numOfTimeslot)