import sys
import copy
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 
import math
import random
import gurobipy as gp
from queue import PriorityQueue
from numpy import log as ln
from random import sample
import numpy as np

import networkx as nx
# ctx._force_start_method('spawn')

sys.path.insert(0, "../../rl")

from SchedulerAgent import SchedulerAgent

EPS = 1e-6
pool = None
class SCHEDULEROUTEGREEDY_CACHE_PS(AlgorithmBase):
    def __init__(self, topo,param=None, name=''):
        super().__init__(topo)
        self.name = name
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        # self.entAgent = DQNAgentDistEnt(self, 0)
        # self.routingAgent = DQRLAgent(self , 0)
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        self.hopCountThreshold = 25
        self.requestState = []
        self.optPaths = {}
        # self.pool = None
        self.schedulerAgent = SchedulerAgent(self , 0)


    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        return (varName + str(parName)).replace(' ', '').replace(',', '][')
    
    def printResult(self):
        self.topo.resetEntanglement()
        self.result.waitingTime = self.totalWaitingTime / self.totalRequest
        self.result.usedQubits = self.totalUsedQubits / self.totalRequest
        
        # self.result.remainRequestPerRound.append(len(self.requests) / self.totalRequest)
        self.result.remainRequestPerRound.append(len(self.requests))
        
        print("[REPS] total time:", self.result.waitingTime)
        print("[REPS] remain request:", len(self.requests))
        print("[REPS] current Timeslot:", self.timeSlot)



        print('[REPS] idle time:', self.result.idleTime)
        print('[' , self.name, '] :' , self.timeSlot, ' total successful request::', self.result.successfulRequest)
        print('[' , self.name, '] :' , self.timeSlot, ' average path len        ::', self.result.pathlen/(self.result.successfulRequest+ 1))
        print('[' , self.name, '] :' , self.timeSlot, ' total path      ::', self.result.totalPath)

        print('[REPS] remainRequestPerRound:', self.result.remainRequestPerRound[-1])
        print('[REPS] avg usedQubits:', self.result.usedQubits)

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
            # print('addnewsdpair ' , len(self.requests) , self.timeSlot)

        self.srcDstPairs = []
        self.requestState = []
        index = 0
        for request in self.requests:
            src = request[0]
            dst = request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))

            self.requestState.append([src,dst , False , [] , index])
            index += 1

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime += len(self.requests)
        # self.tryPreSwapp()

        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.PFT()
            self.randPFT()
            # self.entAgent.learn_and_predict()
        # print('[REPS] p2 end')
    
    def randPFT(self):
        assignable = True
        while assignable:
            assignable = False
            for link in self.topo.links:
                if link.assignable():
                    assignable = True
                    if np.random.random() > 0.5:
                    
                        link.assignQubits()
                        self.totalUsedQubits += 2
    
        
    def get_next_request_id(self):
        if 'RAND' in self.name :
            return self.schedulerAgent.learn_and_predict_next_random_request(self.requestState)
        else:
            return self.schedulerAgent.learn_and_predict_next_request_route(self.requestState)


            
    def scheduleAndAssign(self):
        reqs = copy.deepcopy(self.requestState)
        foundPath = 0
        reqmask = [0 for _ in reqs]
        schedule = []

        t = 0
        successReq = 0
        while len(reqs):
            assign = False
            # print('------------------req len----------------' , len(reqs))
            # print(self.requestState)
            current_state, next_req_id , qval = self.get_next_request_id()
            schedule.append(next_req_id)

            req = self.requestState[next_req_id]
            reqIndex = req[0]
            path = self.get_path(req)
            # print('lenpath ' , len(path))
            if len(path):


                assign = self.assignEntangledPath(path) 
                

            for r in reqs:
                if r[0].id == req[0].id and r[1].id == req[1].id:
                    reqs.remove(r)
                    break
            for r in self.requestState:
                if r[0].id == req[0].id and r[1].id == req[1].id:
                    r[2] = True #checked
                    r[3] = path
                    break
            if assign:
                reward = 1
                foundPath +=1
            else:
                 reward = -1
            reqmask[next_req_id] = 1
            self.schedulerAgent.update_action( self.requestState ,  next_req_id  , current_state , done = (len(reqs) == 0) , t = t , successReq = foundPath,qval = qval,reqmask=copy.deepcopy(reqmask) , reqIndex = reqIndex)
            t+= 1
    
        print('[' , self.name, '] :' , self.timeSlot, ' path found :', foundPath, 'schedule:' , schedule)



    def assignEntangledPath(self , path):
        selectedLinks = []
        for i in range(len(path) -1):
            n1 = self.topo.nodes[path[i]]
            n2 = self.topo.nodes[path[i+1]]
            assigned = 0
            for link in n1.links:
                if link.contains(n2):
                    if link.isEntangled(self.timeSlot) and not link.taken:
                        selectedLinks.append(link)
                        assigned += 1
                        break
            if not assigned:
                return False
        for link in selectedLinks:
            link.taken = True
        return True


    
    def p4(self):
        self.scheduleAndAssign()


        self.ELS()
            
        # print('[REPS] p4 end') 
        self.printResult()
        # self.entAgent.update_reward()
        self.schedulerAgent.update_reward()
        return self.result
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
                    if width == 0:
                        continue
                    failedFindPath = False
                    for nodeIndex in range(pathLen - 1):
                        node = Pi[SDpair][k][nodeIndex]
                        next = Pi[SDpair][k][nodeIndex + 1]
                        self.fi[SDpair][(node, next)] += width

            sorted(paths, key = self.widthForSort)

            for path in paths:
                pathLen = len(path) - 1
                width = path[-1]
                SDpair = (path[0], path[-2])
                isable = True
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    if self.edgeCapacity(node, next) < 1:
                        isable = False
                
                if not isable:
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                    continue
                
                failedFindPath = False
                self.ti[SDpair] += 1
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    self.fi[SDpair][(node, next)] += 1

        # print('[REPS] PFT end')
        for SDpair in self.srcDstPairs:
            for (p , q) in  self.topo.edgeIndices:
                u = self.topo.nodes[p]
                v = self.topo.nodes[q]
                need = self.fi[SDpair][(u, v)] + self.fi[SDpair][(v, u)]
                if need:
                    assignCount = 0
                    for link in u.links:
                        if link.contains(v) and link.assignable():
                            # link(u, v) for u, v in edgeIndices)
                            link.assignQubits()
                            self.totalUsedQubits += 2
                            assignCount += 1
                            if assignCount == need:
                                break
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
    def LP1(self):
        # print('[REPS-CACHE] LP1 start')
        # initialize fi(u, v) ans ti

        self.fi_LP = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti_LP = {SDpair : 0 for SDpair in self.srcDstPairs}
        
        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)

        edgeIndices = []
        notEdge = []
        # for edge in self.topo.edges:
        #     edgeIndices.append((edge[0].id, edge[1].id))

        # print('*** ========+============================== edgeindices len by edge ' , len(edgeIndices))


        edgeIndices = set()
        for link in self.topo.links:
            if (link.n1.id, link.n2.id) not in edgeIndices and (link.n2.id, link.n1.id) not in edgeIndices:
                edgeIndices.add((link.n1.id, link.n2.id))
        # print('========+============================== edgeindices len ' , len(edgeIndices))
        self.topo.edgeIndices = edgeIndices
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))
        # LP

        m = gp.Model('REPS for PFT')
        m.setParam("OutputFlag", 0)
        f = m.addVars(numOfSDpairs, numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "f")
        t = m.addVars(numOfSDpairs, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "t")
        x = m.addVars(numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "x")
        m.update()
        
        m.setObjective(gp.quicksum(t[i] for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            m.addConstr(gp.quicksum(f[i, s, v] for v in range(numOfNodes)) - gp.quicksum(f[i, v, s] for v in range(numOfNodes)) == t[i])

            d = self.srcDstPairs[i][1].id
            m.addConstr(gp.quicksum(f[i, d, v] for v in range(numOfNodes)) - gp.quicksum(f[i, v, d] for v in range(numOfNodes)) == -t[i])

            for u in range(numOfNodes):
                if u not in [s, d]:
                    m.addConstr(gp.quicksum(f[i, u, v] for v in range(numOfNodes)) - gp.quicksum(f[i, v, u] for v in range(numOfNodes)) == 0)

        
        for (u, v) in edgeIndices:
            dis = self.topo.distance(self.topo.nodes[u].loc, self.topo.nodes[v].loc)
            # probability = math.exp(-self.topo.alpha * dis)
            # print('++===+++== orig prob ' , self.timeSlot , probability)
            links = [link for link in self.topo.links if ((link.n1.id == u and link.n2.id == v) or (link.n1.id == v and link.n2.id == u))]
            # print('++++++ ' , len(links))
            
            prob = 0
            isVirtual = False
            for link in links:
                if link.isEntangled(self.timeSlot):
                    prob+=1
                    # if link.isVirtualLink:
                    #     print('====================++++++++++++++++vlinkkkkkkkkkkkk++++++++++++++++++==================')
                else:
                    prob +=link.p()
                isVirtual = isVirtual or link.isVirtualLink
            probability = prob/len(links)
            # print('++===+++== later prob ' , self.timeSlot , probability)

            m.addConstr(gp.quicksum(f[i, u, v] + f[i, v, u] for i in range(numOfSDpairs)) <= probability * x[u, v])

            capacity = self.edgeCapacity(self.topo.nodes[u], self.topo.nodes[v])
            # if isVirtual:
            #     print('for the v link capacity: ', (u,v) , capacity , len(links), probability)
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
            m.addConstr(gp.quicksum(x[n1, n2] for (n1, n2) in edgeContainu) <= self.topo.nodes[u].remainingQubits)

        m.optimize()
        if m.status != 2:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++FAILED++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            exit()
        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]
            # for edge in self.topo.edges:
            #     u = edge[0]
            #     v = edge[1]
            for (p , q) in edgeIndices:
                u = self.topo.nodes[p]
                v = self.topo.nodes[q]
                varName = self.genNameByComma('f', [i, u.id, v.id])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            # for edge in self.topo.edges:
            #     u = edge[1]
            #     v = edge[0]
            for (p , q) in edgeIndices:
                v = self.topo.nodes[p]
                u = self.topo.nodes[q]
                varName = self.genNameByComma('f', [i, u.id, v.id])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for (u, v) in notEdge:
                u = self.topo.nodes[u]
                v = self.topo.nodes[v]
                self.fi_LP[SDpair][(u, v)] = 0
            
            
            varName = self.genNameByComma('t', [i])
            self.ti_LP[SDpair] = m.getVarByName(varName).x
        # print('[REPS-CACHE] LP1 end')
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
    def get_path(self , req):
        G = nx.Graph()
        for node in self.topo.nodes:
            G.add_node(node.id)
        link_counter = {}
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and not link.taken:
                G.add_edge(link.n1.id , link.n2.id )
                try:
                    link_counter[(link.n1.id , link.n2.id)] += link_counter[(link.n1.id , link.n2.id)]
                except:
                    link_counter[(link.n1.id , link.n2.id)] = 1
        
        for edge in G.edges():
            try:
                count = link_counter[(edge[0] , edge[1])]
            except:
                count = link_counter[(edge[1] , edge[0])]

            nx.set_edge_attributes(G, {edge: {"weight": 1/count}})
            
        try:
            path = nx.shortest_path(G , req[0].id , req[1].id ,weight='weight')
        except:
            path = []
        return path

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


    
          
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.isEntangled(self.timeSlot):
                capacity += 1
        # print(capacity)
        return capacity

    



    def ELS(self):

        #work for len 2
        #work for len 2
        #work for len 2
        #work for len 2
        
        
        # print('[REPS] ELS end')
        # print('[REPS]' + [(src.id, dst.id) for (src, dst) in self.srcDstPairs])
        totalEntanglement = 0
        successReq = 0
        usedLinks = set()
        assignedCount = 0
        for SDpair in self.requestState:
            src = SDpair[0]
            dst = SDpair[1]
            path = SDpair[3]
            if len(path):
                assignedCount += 1
        # print('in ELSSSSSSSSSSSSSSSSSSSSSSs assignedCount ' , assignedCount)

        for SDpair in self.requestState:
            src = SDpair[0]
            dst = SDpair[1]
            assigned = SDpair[2]
            # if not assigned:
            #     continue
            path = SDpair[3]
            if not len(path):
                continue
            needLink = []
            
            for i in range(len(path) -2):
                n1 = self.topo.nodes[path[i]]
                n2 = self.topo.nodes[path[i+1]]
                n3 = self.topo.nodes[path[i+2]]
                link1 = None
                link2 = None
                for link in n1.links:
                    if link.contains(n2) and link.isEntangled(self.timeSlot):
                        link1 = link
                        break
                for link in n2.links:
                    if link.contains(n3) and link.isEntangled(self.timeSlot):
                        link2 = link
                        break
                if link1 is not None and link2 is not None:
                    needLink.append((n2 , link1 , link2))
            for node , link1 , link2 in needLink:
                usedLinks.add(link1)
                usedLinks.add(link2)
                swapped = node.attemptSwapping(link1, link2)
                if swapped:
                    self.topo.usedLinks.add(link1)
                    self.topo.usedLinks.add(link2)
            successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)
            for path2 in successPath:
                for node, link in path2:
                    if link is not None:
                        link.used = True
                        edge = self.topo.linktoEdgeSorted(link)

                        self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward


            if len(successPath):
                for request in self.requests:
                    if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                        self.requests.remove(request)
                        successReq += 1
                        break

            for (node, link1, link2) in needLink:
                if not link1 is None and not link1.used and link1.isEntangled(self.timeSlot):
                    edge = self.topo.linktoEdgeSorted(link1)
                        
                    try:
                        self.topo.reward_ent[edge] += self.topo.negative_reward
                    except:
                        self.topo.reward_ent[edge] = self.topo.negative_reward

                if not link2 is None and not link2.used and link2.isEntangled(self.timeSlot):

                    edge = self.topo.linktoEdgeSorted(link2)
                        
                    try:
                        self.topo.reward_ent[edge] += self.topo.negative_reward
                    except:
                        self.topo.reward_ent[edge] = self.topo.negative_reward
                if link1 in self.topo.usedLinks:
                    link1.clearPhase4Swap()
                if link2 in self.topo.usedLinks:
                    link2.clearPhase4Swap()
                
            totalEntanglement += len(successPath)
            print('before update needlink dict ====================')
            self.updateNeedLinksDict([self.topo.nodes[n] for n in path])


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


    
if __name__ == '__main__':
    
    numOfRequestPerRound = 20
    print('----------------calling generate() from main------------')
    topo = Topo.generate(100, 1, 5, 0.0002, 2)
    topo.setNumOfRequestPerRound(numOfRequestPerRound)
    s = SCHEDULEGREEDY(topo,name='SCHEDULEGREEDY')
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
            for j in range(numOfRequestPerRound):
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