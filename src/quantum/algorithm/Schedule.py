import sys
import copy
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 

from numpy import log as ln
from random import sample
import numpy as np

import networkx as nx
# ctx._force_start_method('spawn')

sys.path.insert(0, "../../rl")

from SchedulerAgent import SchedulerAgent

EPS = 1e-6
pool = None
class SCHEDULEGREEDY(AlgorithmBase):
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
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.scheduleAndAssign()
            # self.randPFT()
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
        if self.name == 'SCHEDULEGREEDY':
            return self.schedulerAgent.learn_and_predict_next_request(self.requestState)
        elif self.name == 'RANDSCHEDULEGREEDY':
            return self.schedulerAgent.learn_and_predict_next_random_request(self.requestState)

            
    def scheduleAndAssign(self):
        reqs = copy.deepcopy(self.requestState)
        print('reqss: ' ,[(r[0].id , r[1].id) for r in reqs])

        reqmask = [0 for _ in reqs]
        # schedule = [4, 3, 0, 5, 1, 7, 6, 8, 9, 2]
        # schedule = [7, 1, 5, 3, 4, 6, 2, 0]
        schedule = []
        foundPath = 0
        t = 0
        successReq = 0
        while len(reqs):
            assign = False
            # print('------------------req len----------------' , len(reqs))
            # print(self.requestState)
            current_state, next_req_id , qval = self.get_next_request_id()
            schedule.append(next_req_id)
            # next_req_id = schedule[t]
            req = self.requestState[next_req_id]
            path = self.get_path(req)
            # print('lenpath ' , len(path))
            if len(path):


                assign = self.assignQubitPath(path) 
                

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
            # print('reqmask' , reqmask)
            self.schedulerAgent.update_action( self.requestState ,  next_req_id  , current_state , done = (len(reqs) == 0) , t = t , successReq = foundPath,qval = qval,reqmask=copy.deepcopy(reqmask))
            t+= 1
    
        print('[' , self.name, '] :' , self.timeSlot, ' path found :', foundPath , 'schedule:' , schedule)



    def assignQubitPath(self , path):
        
        for i in range(len(path) -1):
            n1 = self.topo.nodes[path[i]]
            n2 = self.topo.nodes[path[i+1]]
            assigned = 0
            for link in n1.links:
                if link.contains(n2):
                    if link.assignable():
                        link.assignQubits()
                        assigned += 1
                        break
                    else:   
                       return False
            if not assigned:
                return False
        return True


    
    def p4(self):


        self.ELS()
            
        # print('[REPS] p4 end') 
        self.printResult()
        # self.entAgent.update_reward()
        self.schedulerAgent.update_reward()
        return self.result
    
   
    def get_path(self , req):
        G = nx.Graph()
        for node in self.topo.nodes:
            G.add_node(node.id)
        for link in self.topo.links:
            if link.assignable():
                G.add_edge(link.n1.id , link.n2.id , weight=1/min(link.n1.remainingQubits , link.n2.remainingQubits))
        
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
            if link.contains(v) and link.entangled:
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
                node.attemptSwapping(link1, link2)
            successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)
            for path in successPath:
                for node, link in path:
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