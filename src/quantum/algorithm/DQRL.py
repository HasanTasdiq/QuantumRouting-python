import sys
import math
import random
import gurobipy as gp
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 

from numpy import log as ln
from random import sample
import numpy as np
import time


# ctx._force_start_method('spawn')

sys.path.insert(0, "../../rl")

from DQRLAgent import DQRLAgent

EPS = 1e-6
pool = None
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
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        self.hopCountThreshold = 25
        self.requestState = []
        self.optPaths = {}
        # self.pool = None



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
            self.requestState.append([src,dst , src , tuple([src.id]) , index , False])
            index += 1

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime += len(self.requests)
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.PFT() # compute (self.ti, self.fi)
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
    def LP1(self):
        # print('[REPS] LP1 start')
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
        m.setParam("OutputFlag", 0)
        f = m.addVars(numOfSDpairs, numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "f")
        t = m.addVars(numOfSDpairs, lb = 0,ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t")
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
            # dis = self.topo.distance(self.topo.nodes[u].loc, self.topo.nodes[v].loc)
            dis = self.topo.dist[(u , v)]
            probability = math.exp(-self.topo.alpha * dis)
            m.addConstr(gp.quicksum(f[i, u, v] + f[i, v, u] for i in range(numOfSDpairs)) <= probability * x[u, v])

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
            m.addConstr(gp.quicksum(x[n1, n2] for (n1, n2) in edgeContainu) <= self.topo.nodes[u].remainingQubits)

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
   
    def prep4(self):
        if len(self.srcDstPairs) > 0:
            self.Pi = {SDpair : [] for SDpair in self.srcDstPairs}

            for i in range(4):
                self.EPS()
                s = self.ELS(self.Pi)
            
                print('=====---=====prep4 ----=====----===== ' , self.timeSlot , i , s)
                if not s:
                    break

            print('=====---=====prep4 final ----=====----===== ' , self.timeSlot , i , s)
            for sd in self.Pi:
                for path in self.Pi[sd]:
                    print([n.id for n in path])
            
    def p4(self):

        # self.prep4()

        if len(self.srcDstPairs) > 0:
            # self.EPS()
            # self.ELS()
            # if 'greedy_only' in  self.name:
            #     self.route_seq()
            # else:
            #     # self.route()
            #     self.route_all_seq()
            # self.route_seq()
            self.route_schedule_seq()
            
        # print('[REPS] p4 end') 
        self.printResult()
        # self.entAgent.update_reward()
        reward = 0
        if not 'greedy_only' in self.name:
            reward = self.routingAgent.update_reward(self.result.successfulRequestPerRound[-1])
        self.result.rewardPerRound.append(reward)

        return self.result
    def get_action_ILP(self):
        req_index = [req[4] for req in self.requestState if not req[[5]]]
        req = self.requestState[random.choice(req_index)]
        next_node  = self.Pi[req][0][1]
        self.Pi[req][0].pop(0)
        
    def get_all_paths(self):
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        T = []
        for request in  self.requests:
            T.append(request)
        # print(self.name , ('greedy_only' in  self.name))
        # print('srcDstPairs',[(r[0].id , r[1].id) for r in self.srcDstPairs])
        # print('requests', [(r[0].id , r[1].id) for r in self.requests])
        # print('requestState::: ' , [(src.id,dst.id,next_node.id,[p for p in list(path)]) for (src,dst,next_node,path) in self.requestState])


        selectedNodesDict = {}
        selectedEdgesDict = {}
        prevlinksDict = {}
        usedLinksDict = {}
        paths = {}
        conflicts = []



        for reqState in self.requestState:
            src, dst = reqState[0] , reqState[1]
            selectedNodesDict[(src, dst)] = [src]
            selectedEdgesDict[(src, dst)] = []
            usedLinksDict[(src, dst)] = []
            prevlinksDict[(src,dst)] = None
            paths[(src,dst)] = [{'current_node':src, 'current_state': [], 'q_val':0}]





        while len(self.requestState):
            if 'greedy_only' in  self.name:
                break

            # print('start while::: ' , [(src.id,dst.id,next_node.id,[p for p in list(path)]) for (src,dst,next_node,path) in self.requestState])

            reqState_action = self.routingAgent.learn_and_predict_next_node_batch(self.requestState)
            # reqState_action = self.routingAgent.learn_and_predict_next_node_batch_shortest(self.requestState)
            # print('in whileeeeee ' , len(reqState_action))
            for (reqState , next_node_id , q , current_state) in  reqState_action:
                # print('start for:: ', current_state)

                (src , dst , current_node , path,index) = reqState
                path = list(path)
                request = (src,dst)
                selectedEdges = selectedEdgesDict[(src,dst)] 
                selectedNodes = selectedNodesDict[(src,dst)] 
                usedLinks = usedLinksDict[(src, dst)]
                prev_links = prevlinksDict[(src,dst)] 
                if len(selectedNodes) <=1:
                    prev_node = None
                good_to_search = True
                failed_no_ent = False
                failed_loop = False
                failed_swap = False
                fail_hopcount = False
                failed_conflict = False
                success = False

                next_node = self.topo.nodes[next_node_id]
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                
                if not len(ent_links):
                    good_to_search = False
                    failed_no_ent = True
                    # print((src.id,dst.id) , '=FAILED= no ent links')
                    conflicts_ = self.getConflicts(current_node , next_node , selectedEdgesDict)
                    if len(conflicts_):
                        failed_conflict = True
                    conflicts.extend(conflicts_)

                else:
                    ent_links = [ent_links[0]]
                
                # for l in ent_links:
                #     l.taken = True

                # print(current_node.id , next_node_id , len(ent_links))


                if current_node == next_node:
                    good_to_search = False
                    failed_loop = True
                    # print((src.id,dst.id) , '=FAILED= current_node == next_node')

                # print(path , next_node.id)
                if next_node.id in path:
                    good_to_search = False
                    failed_loop = True

                    # print((src.id,dst.id) , '=FAILED= loop')
                    
                # hopCount += 1
                # if hopCount >= self.hopCountThreshold:
                #     fail_hopcount = True
                #     # print((src.id,dst.id) , '=FAILED= hopcount exceeds')
                        
                    
                # if good_to_search:
                #     # dist = current_state[self.routingAgent.env.SIZE *2 +2][next_node_id] + 1
                #     dist = current_state[self.routingAgent.env.SIZE +2][next_node_id] + 1
                #     if not dist:
                #         dist = 1
                #     try:
                #         self.topo.reward_routing[key] += self.topo.positive_reward / dist
                #     except:
                #         self.topo.reward_routing[key] = self.topo.positive_reward / dist
                # else:
                #     if failed_no_ent or failed_loop:
                #         try:
                #             self.topo.reward_routing[key] += self.topo.negative_reward
                #         except:
                #             self.topo.reward_routing[key] = self.topo.negative_reward
                        
                # if prev_node is not None:
                #     if good_to_search:
                #         swapCount = 0
                #         swappedlinks = []
                #         for link1,link2 in zip(prev_links , ent_links):
                #             swapped = current_node.attemptSwapping(link1, link2)
                #             if swapped:
                #                 swappedlinks.append(link2)
                #             usedLinks.append(link2)
                #         if len(swappedlinks):
                #             prev_links = swappedlinks
                #             prevlinksDict[(src,dst)] = prev_links
                #         else:
                #             good_to_search = False
                #             failed_swap = True
                #             # print((src.id,dst.id) , '=FAILED= swap fails')


                prev_links = ent_links
                prevlinksDict[(src,dst)] = prev_links

                usedLinks.extend(prev_links)
                    

                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))
                selectedEdgesDict[(src,dst)] = selectedEdges
                selectedNodesDict[(src,dst)]  = selectedNodes
                usedLinksDict[(src, dst)] = usedLinks
                prevlinksDict[(src,dst)] = prev_links

                
                path.append(next_node.id)
                paths[(src,dst)].append({'current_node': next_node , 'current_state' : current_state, 'q_val': q})
                if not good_to_search:
                    paths[(src,dst)] = []
                    
                if len(prev_links) and next_node == request[1] and good_to_search:
                    success = True
                    good_to_search = False
                
                    
                # self.routingAgent.update_action( request ,  next_node_id  , current_state , path , success)

                        
                prev_node = current_node
                current_node = next_node
                # print([(src.id,dst.id,next_node.id,[p for p in list(path)]) for (src,dst,next_node,path) in self.requestState])
                # print(reqState[0].id,reqState[1].id,reqState[2].id, [p for p in list(reqState[3])])
                
                self.requestState.remove(reqState)
                reqState = (src,dst,current_node,tuple(path), index)
                if not success and good_to_search:
                    self.requestState.append(reqState)
                
                
        
        
        tp = 0
        pathlen = 0
        for key in paths:
            print(key[0].id , key[1].id)
            print([n['current_node'].id for n in paths[key] ])
            pathlen += len(paths[key])
            if len(paths[key]) > 0:
                tp += 1
        # if tp > 0 :
        #     self.result.pathlen += pathlen
        self.result.totalPath += tp
        
        print('===== paths found =======================================' , tp)

            # print([n['current_state'] for n in paths[key] ])

            # for n in paths[key]:
            #     print('========= ' ,n['current_node'].id , '==============')
            #     print(n['current_state'])


        return paths

        for request , current_node , next_node in conflicts:
            neg_weight = 0.05
            key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

            try:
                self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
            except:
                self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight

        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request  after  extra:', successReq)
    def route_all_seq(self ):
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        conflicts = []
        selectedEdgesDict = {}
        paths = self.get_all_paths()

        pathlen = 0

        T = self.routingAgent.getOrderedRequests(paths)
        # T = []
        # for request in  self.requests:
        #     T.append(request)
        # print(self.name , ('greedy_only' in  self.name))
        # print([(r[0].id , r[1].id) for r in self.srcDstPairs])
        # print([(r[0].id , r[1].id) for r in self.requests])
        # print([(r[0].id , r[1].id) for r in T])
        # if not (self.param is not None and 'greedy_only' in self.param):
        if True:
            for request in T:
                if 'greedy_only' in  self.name:
                    continue

                # print('========ent_links=======')
                # print((request[0].id, request[1].id))

                # if not len(self.findDQRLPath(request)):
                #     continue

                src,dst = request[0] , request[1]
                targetPath = paths[(src , dst)]
                if not len(targetPath):
                    continue

                current_node = request[0]
                current_state = targetPath[1]['current_state']

                prev_node = None
                prev_links = []
                hopCount = 0
                success = False
                good_to_search = True
                width = 0
                usedLinks = []
                selectedNodes = []
                selectedEdges = []
                path = [current_node.id]
                failed_no_ent = False
                failed_loop = False
                failed_swap = False
                fail_hopcount = False
                failed_conflict = False
                i = 1
                # targetPath = self.findPathForDQRL((src,dst))

                while (not current_node == request[1]) and (hopCount < self.hopCountThreshold) and good_to_search:
                    
                    current_node_id = current_node.id
                    next_node = targetPath[i]['current_node']
                    next_state = targetPath[(i+1)%len(targetPath)]['current_state']
                    next_node_id = next_node.id
                    i+= 1
                    ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
                    key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                    if not len(ent_links):
                        good_to_search = False
                        failed_no_ent = True
                        # print((src.id,dst.id) , '=FAILED= no ent links')
                        conflicts_ = self.getConflicts(current_node , next_node , selectedEdgesDict)
                        if len(conflicts_):
                            failed_conflict = True
                        conflicts.extend(conflicts_)

                    else:
                        ent_links = [ent_links[0]]

                    # print(current_node.id , next_node_id , len(ent_links))


                    if current_node == next_node:
                        good_to_search = False
                        failed_loop = True
                        # print((src.id,dst.id) , '=FAILED= current_node == next_node')

                        
                    if next_node.id in path:
                        good_to_search = False
                        failed_loop = True

                        # print((src.id,dst.id) , '=FAILED= loop')
                    
                    hopCount += 1
                    if hopCount >= self.hopCountThreshold:
                        fail_hopcount = True
                        # print((src.id,dst.id) , '=FAILED= hopcount exceeds')
                        
                    
                    if good_to_search:
                        # dist = current_state[self.routingAgent.env.SIZE *2 +2][next_node_id] + 1
                        dist = current_state[self.routingAgent.env.SIZE +2][next_node_id] + 1
                        if not dist:
                            dist = 1
                        try:
                            self.topo.reward_routing[key] += self.topo.positive_reward / dist
                        except:
                            self.topo.reward_routing[key] = self.topo.positive_reward / dist
                    else:
                        if failed_no_ent or failed_loop:
                            try:
                                self.topo.reward_routing[key] += self.topo.negative_reward
                            except:
                                self.topo.reward_routing[key] = self.topo.negative_reward
                        if failed_conflict:
                            try:
                                self.topo.reward_routing[key] += self.topo.negative_reward
                            except:
                                self.topo.reward_routing[key] = self.topo.negative_reward
                        
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
                                failed_swap = True
                                # print((src.id,dst.id) , '=FAILED= swap fails')

                                
                    else:
                        prev_links = ent_links
                        usedLinks.extend(prev_links)
                    

                    selectedNodes.append(next_node)
                    selectedEdges.append((current_node, next_node))
                    path.append(next_node.id)
                    
                    if len(prev_links) and next_node == request[1] and good_to_search:
                        success = True
                        good_to_search = False
                    
                    self.routingAgent.update_action( request ,current_node.id ,   next_node_id  , current_state , path , success)

                        
                    prev_node = current_node
                    current_node = next_node
                    current_state = next_state


                    
                # s = min(src.id , dst.id)
                # d = max(src.id , dst.id)
                # try:
                #     self.topo.pair_dict[(s,d)] += 1
                # except:
                #     self.topo.pair_dict[(s,d)] = 1

                if success:
                    pathlen += len(path)

                    successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)

                    for path_ in successPath:
                        for node, link in path_:
                            if link is not None:
                                link.used = True
                                edge = self.topo.linktoEdgeSorted(link)

                                self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward
                            # if node is not None:
                            #     try:
                            #         self.topo.reward_routing[(request , node)] += self.topo.positive_reward
                            #     except:
                            #         self.topo.reward_routing[(request , node)] = self.topo.positive_reward
                            #                     self.topo.reward_ent[edge] = self.topo.negative_reward

                    for (current_node, next_node) in selectedEdges:
                        key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                        # print(key)
                        try:
                            self.topo.reward_routing[key] += self.topo.positive_reward *2
                        except:
                            self.topo.reward_routing[key] = self.topo.positive_reward *2
                        # print(key , '  ==  ' , self.topo.reward_routing[key])
                        
                    print("!!!!!!!success!!!!!!!" , src.id , dst.id , [n for n in path])
                    # print('shortest path ----- ' , [n.id for n in targetPath])
                    

                    # print([(r[0].id, r[1].id) for r in self.requests])

                    # self.requests.remove(request)
                    for req in self.requests:
                        src = req[0]
                        dst = req[1]
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(req)
                            break

                    successReq += 1
                    totalEntanglement += len(successPath)
                    
                else:
                    print("!!!!!!!fail!!!!!!!" , src.id , dst.id , [n for n in path])
                    # print('shortest path ----- ' , [n.id for n in targetPath])
                    print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                    for link in usedLinks:
                            edge = self.topo.linktoEdgeSorted(link)
                            try:
                                self.topo.reward_ent[edge] += self.topo.negative_reward
                            except:
                                self.topo.reward_ent[edge] = self.topo.negative_reward
                    
                    
                    neg_weight = 0
                    # if failed_no_ent:
                    #     neg_weight = 10
                    # elif failed_loop:
                    #     neg_weight = 12
                    # elif failed_swap:
                    #     neg_weight = 0
                    # elif fail_hopcount:
                    #     neg_weight = 10
                    
                    if failed_no_ent:
                        neg_weight = 0.05
                    elif failed_loop:
                        neg_weight = 0.05
                    elif failed_swap:
                        neg_weight = 0
                    elif fail_hopcount:
                        neg_weight = 0.05
                    
                    if failed_conflict:
                        neg_weight = 0.1

                    for (current_node, next_node) in selectedEdges:
                        key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                        try:
                            self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
                        except:
                            self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight
                        
                        # print(key , '  ==  ' , self.topo.reward_routing[key])
                selectedEdgesDict[(src,dst)] = selectedEdges
                for link in usedLinks:
                    link.clearPhase4Swap()

        for request , current_node , next_node in conflicts:
            neg_weight = 0.05
            key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

            try:
                self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
            except:
                self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight

        self.result.pathlen += pathlen

        self.result.usedLinks += len(usedLinks)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request before extra:', successReq)

        self.result.conflicts += len(conflicts)
        print('[' , self.name, '] :' , self.timeSlot,'========total=conflicts========== ' , self.result.conflicts)


        extra_successReq , extra_totalEntanglement = 0 , 0
        # if 'greedy_only' in self.name:
        #     extra_successReq , extra_totalEntanglement = self.extraRoute()

        totalEntanglement += extra_totalEntanglement
        successReq += extra_successReq
        
        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request  after  extra:', successReq)
    
    

    def route(self):
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        T = []
        for request in  self.requests:
            T.append(request)
        # print(self.name , ('greedy_only' in  self.name))
        # print('srcDstPairs',[(r[0].id , r[1].id) for r in self.srcDstPairs])
        # print('requests', [(r[0].id , r[1].id) for r in self.requests])
        # print('requestState::: ' , [(src.id,dst.id,next_node.id,[p for p in list(path)]) for (src,dst,next_node,path) in self.requestState])


        selectedNodesDict = {}
        selectedEdgesDict = {}
        prevlinksDict = {}
        usedLinksDict = {}

        for reqState in self.requestState:
            src, dst = reqState[0] , reqState[1]
            selectedNodesDict[(src, dst)] = [src]
            selectedEdgesDict[(src, dst)] = []
            usedLinksDict[(src, dst)] = []
            prevlinksDict[(src,dst)] = None
        conflicts = []





        while len(self.requestState):
            if 'greedy_only' in  self.name:
                break

            # print('start while::: ' , [(src.id,dst.id,next_node.id,[p for p in list(path)]) for (src,dst,next_node,path) in self.requestState])

            reqState_action = self.routingAgent.learn_and_predict_next_node_batch(self.requestState)
            # print('in whileeeeee ' , len(reqState_action))
            for (reqState , next_node_id , q , current_state) in  reqState_action:
                # print('start for:: ', current_state)

                (src , dst , current_node , path) = reqState
                path = list(path)
                request = (src,dst)
                selectedEdges = selectedEdgesDict[(src,dst)] 
                selectedNodes = selectedNodesDict[(src,dst)] 
                usedLinks = usedLinksDict[(src, dst)]
                prev_links = prevlinksDict[(src,dst)] 
                if len(selectedNodes) <=1:
                    prev_node = None
                good_to_search = True
                failed_no_ent = False
                failed_loop = False
                failed_swap = False
                fail_hopcount = False
                failed_conflict = False
                success = False

                next_node = self.topo.nodes[next_node_id]
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                
                if not len(ent_links):
                    good_to_search = False
                    failed_no_ent = True
                    # print((src.id,dst.id) , '=FAILED= no ent links')
                    conflicts_ = self.getConflicts(current_node , next_node , selectedEdgesDict)
                    if len(conflicts_):
                        failed_conflict = True
                    conflicts.extend(conflicts_)

                else:
                    ent_links = [ent_links[0]]
                
                for l in ent_links:
                    l.taken = True

                # print(current_node.id , next_node_id , len(ent_links))


                if current_node == next_node:
                    good_to_search = False
                    failed_loop = True
                    # print((src.id,dst.id) , '=FAILED= current_node == next_node')

                        
                if next_node.id in path:
                    good_to_search = False
                    failed_loop = True

                    # print((src.id,dst.id) , '=FAILED= loop')
                    
                # hopCount += 1
                # if hopCount >= self.hopCountThreshold:
                #     fail_hopcount = True
                #     # print((src.id,dst.id) , '=FAILED= hopcount exceeds')
                        
                    
                if good_to_search:
                    # dist = current_state[self.routingAgent.env.SIZE *2 +2][next_node_id] + 1
                    dist = current_state[self.routingAgent.env.SIZE +2][next_node_id] + 1
                    if not dist:
                        dist = 1
                    try:
                        self.topo.reward_routing[key] += self.topo.positive_reward / dist
                    except:
                        self.topo.reward_routing[key] = self.topo.positive_reward / dist
                else:
                    if failed_no_ent or failed_loop:
                        try:
                            self.topo.reward_routing[key] += self.topo.negative_reward
                        except:
                            self.topo.reward_routing[key] = self.topo.negative_reward
                        
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
                            prevlinksDict[(src,dst)] = prev_links
                        else:
                            good_to_search = False
                            failed_swap = True
                            # print((src.id,dst.id) , '=FAILED= swap fails')

                                
                else:
                    prev_links = ent_links
                    usedLinks.extend(prev_links)
                    

                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))
                selectedEdgesDict[(src,dst)] = selectedEdges
                selectedNodesDict[(src,dst)]  = selectedNodes
                usedLinksDict[(src, dst)] = usedLinks
                prevlinksDict[(src,dst)] = prev_links

                
                path.append(next_node.id)
                    
                if len(prev_links) and next_node == request[1] and good_to_search:
                    success = True
                    good_to_search = False
                    
                self.routingAgent.update_action( request ,current_node.id,  next_node_id  , current_state , path , success)

                        
                prev_node = current_node
                current_node = next_node
                # print([(src.id,dst.id,next_node.id,[p for p in list(path)]) for (src,dst,next_node,path) in self.requestState])
                # print(reqState[0].id,reqState[1].id,reqState[2].id, [p for p in list(reqState[3])])
                
                self.requestState.remove(reqState)
                reqState = (src,dst,current_node,tuple(path))
                if not success and good_to_search:
                    self.requestState.append(reqState)
                
                if success:
                    successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)

                    for path_ in successPath:
                        for node, link in path_:
                            if link is not None:
                                link.used = True
                                edge = self.topo.linktoEdgeSorted(link)

                                self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward
                            # if node is not None:
                            #     try:
                            #         self.topo.reward_routing[(request , node)] += self.topo.positive_reward
                            #     except:
                            #         self.topo.reward_routing[(request , node)] = self.topo.positive_reward
                            #                     self.topo.reward_ent[edge] = self.topo.negative_reward

                    for (current_node, next_node) in selectedEdges:
                        key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                        # print(key)
                        try:
                            self.topo.reward_routing[key] += self.topo.positive_reward *2
                        except:
                            self.topo.reward_routing[key] = self.topo.positive_reward *2
                        # print(key , '  ==  ' , self.topo.reward_routing[key])
                        
                    print("!!!!!!!success!!!!!!!" , src.id , dst.id , [n for n in path])
                    # print('shortest path ----- ' , [n.id for n in targetPath])
                    

                    # print([(r[0].id, r[1].id) for r in self.requests])

                    # self.requests.remove(request)
                    for req in self.requests:
                        src = req[0]
                        dst = req[1]
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(req)
                            break

                    successReq += 1
                    totalEntanglement += len(successPath)
                    for link in usedLinks:
                        link.clearPhase4Swap()
                        link.taken = False
                elif not good_to_search:
                    print("!!!!!!!fail!!!!!!!" , src.id , dst.id , [n for n in path])
                    # print('shortest path ----- ' , [n.id for n in targetPath])
                    print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                    for link in usedLinks:
                            edge = self.topo.linktoEdgeSorted(link)
                            try:
                                self.topo.reward_ent[edge] += self.topo.negative_reward
                            except:
                                self.topo.reward_ent[edge] = self.topo.negative_reward
                    
                    
                    neg_weight = 0
                    # if failed_no_ent:
                    #     neg_weight = 10
                    # elif failed_loop:
                    #     neg_weight = 12
                    # elif failed_swap:
                    #     neg_weight = 0
                    # elif fail_hopcount:
                    #     neg_weight = 10
                    
                    if failed_no_ent:
                        neg_weight = 0.05
                    elif failed_loop:
                        neg_weight = 0.05
                    elif failed_swap:
                        neg_weight = 0
                    elif fail_hopcount:
                        neg_weight = 0.05

                    for (current_node, next_node) in selectedEdges:
                        key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                        try:
                            self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
                        except:
                            self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight
                    
                    for link in usedLinks:
                        link.clearPhase4Swap()
                        link.taken = False

                #wip
        
        print('==================================================================conflicts========== ' , len(conflicts))
        
        for request , current_node , next_node in conflicts:
            neg_weight = 0.05
            key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

            try:
                self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
            except:
                self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight

        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request  after  extra:', successReq)


    def getConflicts(self, current_node , next_node , selectedEdgesDict):
        conflicts = []
        for req in selectedEdgesDict:
            edges = selectedEdgesDict[req]
            if (current_node, next_node) in edges:
                conflicts.append((req , current_node, next_node))
            elif (next_node, current_node) in edges:
                conflicts.append((req , next_node, current_node))
        
        return conflicts


    def route_seq(self):
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        conflicts = []
        selectedEdgesDict = {}

        T = []
        for request in  self.requests:
            T.append(request)
        # print(self.name , ('greedy_only' in  self.name))
        # print([(r[0].id , r[1].id) for r in self.srcDstPairs])
        print([(r[0].id , r[1].id) for r in self.requests])
        # if not (self.param is not None and 'greedy_only' in self.param):
        if True:
            for request in T:
                for req in self.requestState:
                    if req[0].id == request[0].id and req[1].id == request[1].id:
                        req[5] = True
                done_episode = not len([r for r in self.requestState if not r[5]])
                if 'greedy_only' in  self.name:
                    continue

                # print('========ent_links=======')
                # print((request[0].id, request[1].id))

                # if not len(self.findDQRLPath(request)):
                #     continue

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
                selectedlinks = []
                path = [current_node.id]
                failed_no_ent = False
                failed_loop = False
                failed_swap = False
                fail_hopcount = False
                targetPath = self.findPathForDQRL((src,dst))
                skipRequest = False
                # if not len(targetPath):
                #     continue
                # targetPath = []

                while (not current_node == request[1]) and (hopCount < self.hopCountThreshold) and good_to_search:
                    
                    current_state, next_node_id = self.routingAgent.learn_and_predict_next_node(request, current_node , path)
                    # print('start for:: ', current_state)
                    if next_node_id == len(self.topo.nodes) or not len(targetPath):
                        skipRequest = True
                        self.routingAgent.update_action( request ,request[0].id,  len(self.topo.nodes)  , current_state , path , done_episode)

                        break
                    
                    next_node = self.topo.nodes[next_node_id]
                    ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
                    key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                    if not len(ent_links):
                        good_to_search = False
                        failed_no_ent = True
                        conflicts_ = self.getConflicts(current_node , next_node , selectedEdgesDict)
                        if len(conflicts_):
                            failed_conflict = True
                        conflicts.extend(conflicts_)
                        # print((src.id,dst.id) , '=FAILED= no ent links')
                    else:
                        ent_links = [ent_links[0]]
                        for link in ent_links:
                            link.taken = True

                    # print(current_node.id , next_node_id , len(ent_links))


                    if current_node == next_node:
                        good_to_search = False
                        failed_loop = True
                        # print((src.id,dst.id) , '=FAILED= current_node == next_node')

                        
                    if next_node.id in path:
                        good_to_search = False
                        failed_loop = True

                        # print((src.id,dst.id) , '=FAILED= loop')
                    
                    hopCount += 1
                    if hopCount >= self.hopCountThreshold:
                        fail_hopcount = True
                        # print((src.id,dst.id) , '=FAILED= hopcount exceeds')
                        
                    
                    # if good_to_search:
                    #     # dist = current_state[self.routingAgent.env.SIZE *2 +2][next_node_id] + 1
                    #     dist = current_state[self.routingAgent.env.SIZE +2][next_node_id] + 1
                    #     if not dist:
                    #         dist = 1
                    #     try:
                    #         self.topo.reward_routing[key] += self.topo.positive_reward / dist
                    #     except:
                    #         self.topo.reward_routing[key] = self.topo.positive_reward / dist
                    # else:
                    #     if failed_no_ent or failed_loop:
                    #         try:
                    #             self.topo.reward_routing[key] += self.topo.negative_reward
                    #         except:
                    #             self.topo.reward_routing[key] = self.topo.negative_reward
                        
                    # if prev_node is not None:
                    #     if good_to_search:
                    #         swapCount = 0
                    #         swappedlinks = []
                    #         for link1,link2 in zip(prev_links , ent_links):
                    #             swapped = current_node.attemptSwapping(link1, link2)
                    #             if swapped:
                    #                 swappedlinks.append(link2)
                    #             usedLinks.append(link2)
                    #         if len(swappedlinks):
                    #             prev_links = swappedlinks
                    #         else:
                    #             good_to_search = False
                    #             failed_swap = True
                    #             # print((src.id,dst.id) , '=FAILED= swap fails')

                                
                    # else:
                    #     prev_links = ent_links
                    #     usedLinks.extend(prev_links)
                    prev_links = ent_links
                    if good_to_search:
                        selectedlinks.append(ent_links[0])
                    

                    selectedNodes.append(next_node)
                    selectedEdges.append((current_node, next_node))
                    path.append(next_node.id)
                    
                    if len(prev_links) and next_node == request[1] and good_to_search:
                        success = True
                        good_to_search = False
                    
                    self.routingAgent.update_action( request ,current_node.id,  next_node_id  , current_state , path , ((next_node == request[1] )and done_episode))

                        
                    prev_node = current_node
                    current_node = next_node

                    
                # s = min(src.id , dst.id)
                # d = max(src.id , dst.id)
                # try:
                #     self.topo.pair_dict[(s,d)] += 1
                # except:
                #     self.topo.pair_dict[(s,d)] = 1
                if success:
                    print('going to swap for ' , (src.id , dst.id ))
                    for i in range(len(selectedlinks)-1):
                        l1 = selectedlinks[i]
                        l2 = selectedlinks[i+1]
                        n1 = l1.n1
                        n2 = l1.n2
                        n = n1 if l2.contains(n1) else n2
                        swapped = n.attemptSwapping(l1,l2)

                        usedLinks.append(l1)
                        usedLinks.append(l2)
                        if not swapped:
                            failed_swap = True
                            break
                else:
                    for link in selectedlinks:
                        link.taken = False

                successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)
                if success and len(successPath):

                    for path_ in successPath:
                        for node, link in path_:
                            if link is not None:
                                link.used = True
                                edge = self.topo.linktoEdgeSorted(link)

                                self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward
                            # if node is not None:
                            #     try:
                            #         self.topo.reward_routing[(request , node)] += self.topo.positive_reward
                            #     except:
                            #         self.topo.reward_routing[(request , node)] = self.topo.positive_reward
                            #                     self.topo.reward_ent[edge] = self.topo.negative_reward
                        break

                        
                    print("====success====" , src.id , dst.id , [n for n in path])
                    print('shortest path ----- ' , [n.id for n in targetPath])
                    

                    # print([(r[0].id, r[1].id) for r in self.requests])

                    # self.requests.remove(request)
                    for req in self.requests:
                        src = req[0]
                        dst = req[1]
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(req)
                            break

                    successReq += 1
                    totalEntanglement += len(successPath)
                    
                else:
                    print("!!!!!!!fail!!!!!!!" , src.id , dst.id , [n for n in path])
                    print('shortest path ----- ' , [n.id for n in targetPath])
                    print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                    for link in usedLinks:
                            edge = self.topo.linktoEdgeSorted(link)
                            try:
                                self.topo.reward_ent[edge] += self.topo.negative_reward
                            except:
                                self.topo.reward_ent[edge] = self.topo.negative_reward
                    
                    
                    neg_weight = 0
                    # if failed_no_ent:
                    #     neg_weight = 10
                    # elif failed_loop:
                    #     neg_weight = 12
                    # elif failed_swap:
                    #     neg_weight = 0
                    # elif fail_hopcount:
                    #     neg_weight = 10
                    
                    if failed_no_ent:
                        neg_weight = 0.05
                    elif failed_loop:
                        neg_weight = 0.05
                    elif failed_swap:
                        neg_weight = 0
                    elif fail_hopcount:
                        neg_weight = 0.05

                    # for (current_node, next_node) in selectedEdges:
                    #     key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                    #     try:
                    #         self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
                    #     except:
                    #         self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight
                        
                        # print(key , '  ==  ' , self.topo.reward_routing[key])
                if skipRequest:
                    key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(request[0].id) + '_' + str(len(self.topo.nodes))
                    print(key)
                    reward = 1
                    self.topo.reward_routing[key] = [reward , successReq]



                for (current_node, next_node) in selectedEdges:
                    key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                    # print(key)
                    pathlen = len(path)
                    # reward = successReq/pathlen
                    reward = 1 if len(successPath) else 0
                    try:
                        self.topo.reward_routing[key] += [reward , successReq]
                        print('=====self.topo.reward_routing[key] += successReq====' , key)

                    except:
                        # print('=====self.topo.reward_routing[key] = successReq====...' , key , reward)

                        self.topo.reward_routing[key] = [reward , successReq]
                        # print(key , '  ==  ' , self.topo.reward_routing[key])

                selectedEdgesDict[(src,dst)] = selectedEdges
                
                for link in usedLinks:
                    link.clearPhase4Swap()
        # for req in selectedEdgesDict:
        #     selectedEdges = selectedEdgesDict[req]
        #     for (current_node, next_node) in selectedEdges:
        #         key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

        #         try:
        #             self.topo.reward_routing[key] += successReq
        #         except:
        #             self.topo.reward_routing[key] = successReq 

        # for request , current_node , next_node in conflicts:
        #     neg_weight = 0.05
        #     key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

        #     try:
        #         self.topo.reward_routing[key] += self.topo.negative_reward * neg_weight
        #     except:
        #         self.topo.reward_routing[key] = self.topo.negative_reward * neg_weight

        self.result.usedLinks += len(usedLinks)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request before extra:', successReq)
        print('[' , self.name, '] :' , self.timeSlot, ' =================conflicts :', len(conflicts))


        extra_successReq , extra_totalEntanglement = 0 , 0
        if 'greedy_only' in self.name or 'bruteforce' in self.name:
            extra_successReq , extra_totalEntanglement = self.extraRoute()

        totalEntanglement += extra_totalEntanglement
        successReq += extra_successReq
        
        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request  after  extra:', successReq)




    def route_schedule_seq(self):
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        conflicts = []
  

        T = []
        for request in  self.requests:
            T.append(request)
        
        selectedNodesDict = {}
        selectedEdgesDict = {}
        prevlinksDict = {}
        usedLinksDict = {}
        selectedlinksDict = {}

        for reqState in self.requestState:
            src, dst = reqState[0] , reqState[1]
            selectedNodesDict[(src, dst)] = [src]
            selectedEdgesDict[(src, dst)] = []
            selectedlinksDict[(src,dst)] = []
            usedLinksDict[(src, dst)] = []
            prevlinksDict[(src,dst)] = None

        conflicts = []

        # print(self.name , ('greedy_only' in  self.name))
        # print([(r[0].id , r[1].id) for r in self.srcDstPairs])
        print([(r[0].id , r[1].id) for r in self.requests])
        # if not (self.param is not None and 'greedy_only' in self.param):

        if True:
            while len(T):
                
                current_state , action = self.routingAgent.learn_and_predict_next_req_node()
                # print('req iddddd ' , req_id , next_node_id)
                # print([(req[0].id , req[1].id , req[2].id, req[4]) for req in self.requestState])
                req_id , next_node_id = self.routingAgent.decode_schdeule_route_action(action)

                reqState = self.requestState[req_id]

                (src , dst , current_node , path, index , checked) = reqState

                request = (src , dst)

                # src,dst = request[0] , request[1]
                # current_node = request[0]
                prev_node = None
                prev_links = prevlinksDict[(src,dst)]
                hopCount = 0
                success = False
                good_to_search = True
                width = 0
                usedLinks = usedLinksDict[(src, dst)]
                selectedNodes = selectedNodesDict[(src, dst)]
                selectedEdges = selectedEdgesDict[(src, dst)]
                selectedlinks = selectedlinksDict[(src,dst)] 
                path = list(path)
                failed_no_ent = False
                failed_loop = False
                failed_swap = False
                fail_hopcount = False
                targetPath = self.findPathForDQRL((current_node,dst))
                if not len(targetPath):
                    good_to_search = False
                # targetPath = []
                skipRequest = False
                # if not len(targetPath):
                #     continue
                # targetPath = []

                    

                    
                next_node = self.topo.nodes[next_node_id]
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                ent_links_count = len(ent_links)
                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                if not len(ent_links):
                    good_to_search = False
                    failed_no_ent = True
                    conflicts_ = self.getConflicts(current_node , next_node , selectedEdgesDict)
                    if len(conflicts_):
                        failed_conflict = True
                    conflicts.extend(conflicts_)
                        # print((src.id,dst.id) , '=FAILED= no ent links')
                else:
                    ent_links = [ent_links[0]]
                    for link in ent_links:
                        link.taken = True

                    # print(current_node.id , next_node_id , len(ent_links))


                if current_node == next_node:
                    good_to_search = False
                    failed_loop = True
                    print((src.id,dst.id) , '=FAILED= current_node == next_node')

                        
                if next_node.id in path:
                    good_to_search = False
                    failed_loop = True

                    print((src.id,dst.id) , '=FAILED= loop')
                    

                        

                prev_links = ent_links
                if good_to_search:
                    selectedlinks.append(ent_links[0])
                    

                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))

                selectedEdgesDict[(src,dst)] = selectedEdges
                selectedNodesDict[(src,dst)]  = selectedNodes
                selectedlinksDict[(src,dst)] = selectedlinks
                usedLinksDict[(src, dst)] = usedLinks
                prevlinksDict[(src,dst)] = prev_links
                path.append(next_node.id)
                    
                if len(prev_links) and next_node == request[1] and good_to_search:
                    success = True
                    good_to_search = False
                
                done_episode = (not good_to_search or success) and (len(T)==1)

                self.routingAgent.update_action( request ,current_node.id,  action  , current_state  , done_episode)

                        
                prev_node = current_node
                current_node = next_node
                req_done = (not good_to_search) or success
                # print('good_to_search ' , good_to_search , 'req_done ' , req_done)
                reqState = (src,dst,current_node,tuple(path),index,req_done)
                self.requestState[index] = reqState

                if success:
                    print('going to swap for ' , (src.id , dst.id ))
                    for i in range(len(selectedlinks)-1):
                        l1 = selectedlinks[i]
                        l2 = selectedlinks[i+1]
                        n1 = l1.n1
                        n2 = l1.n2
                        n = n1 if l2.contains(n1) else n2
                        swapped = n.attemptSwapping(l1,l2)

                        usedLinks.append(l1)
                        usedLinks.append(l2)
                        if not swapped:
                            failed_swap = True
                            print('================failed swap==================')
                            break
             


                if success:
                    print('going to find path for:', (src.id , dst.id ))
                    successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)
                    print('len success ' , len(successPath))
                    if len(successPath):

                        for path_ in successPath:
                            for node, link in path_:
                                if link is not None:
                                    link.used = True
                                    edge = self.topo.linktoEdgeSorted(link)

                                    self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward
                
                            break

                        for req in self.requests:
                            # src = req[0]
                            # dst = req[1]
                            if (src, dst) == (req[0], req[1]):
                                # print('[REPS] finish time:', self.timeSlot - request[2])
                                self.requests.remove(req)
                                break

                        successReq += 1
                        totalEntanglement += len(successPath)
                    
                # else:

                #     # for link in usedLinks:
                #     #         edge = self.topo.linktoEdgeSorted(link)
                #     #         try:
                #     #             self.topo.reward_ent[edge] += self.topo.negative_reward
                #     #         except:
                #     #             self.topo.reward_ent[edge] = self.topo.negative_reward
                    
                    


                if ent_links_count:
                    reward = -1/ent_links_count
                else:
                    reward = -1
                if req_done:
                    for req in T:
                        # print('(src, dst) == (req[0], req[1])' , src.id , dst.id , req[0].id , req[1].id , (src, dst) == (req[0], req[1]) , len(T))
                        if (src, dst) == (req[0], req[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            T.remove(req)
                            break
                    if success:
                        print("====success====" , src.id , dst.id , [n for n in path])
                        print('shortest path ----- ' , [n.id for n in targetPath])
                        reward = 10
                        # reward = 1
                    else:
                        for link in selectedlinks:
                            link.taken = False
                        print("!!!!!!!=fail=!!!!!!!" , src.id , dst.id , [n for n in path])
                        print('shortest path ----- ' , [n.id for n in targetPath])
                        print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                        reward = -5
                        # reward = -1
                    # for (current_node, next_node) in selectedEdges:
                    #     key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                    #     try:
                    #         self.topo.reward_routing[key] += reward
                    #     except:
                    #         self.topo.reward_routing[key] = reward

                # print('lenT ' , len(T))

                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(prev_node.id) + '_' + str(next_node.id)

                # reward = -self.topo.numOfRequestPerRound

                try:
                    self.topo.reward_routing[key] += reward
                except:
                    self.topo.reward_routing[key] = reward    

                for link in usedLinks:
                    link.clearPhase4Swap()
                # break
                # time.sleep(.1)


        self.result.usedLinks += len(usedLinks)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request before extra:', successReq)
        print('[' , self.name, '] :' , self.timeSlot, ' =================conflicts :', len(conflicts))


        extra_successReq , extra_totalEntanglement = 0 , 0
        if 'greedy_only' in self.name or 'bruteforce' in self.name:
            extra_successReq , extra_totalEntanglement = self.extraRoute()

        totalEntanglement += extra_totalEntanglement
        successReq += extra_successReq
        
        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)

        self.result.successfulRequest += successReq
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request  after  extra:', successReq)
    


        
    # def findDQRLPath(self , request):
    #         src,dst = request[0] , request[1]
    #         current_node = request[0]
    #         prev_node = None
    #         prev_links = []
    #         hopCount = 0
    #         success = False
    #         good_to_search = True
    #         width = 0
    #         usedLinks = []
    #         selectedNodes = [current_node]
    #         selectedEdges = []
    #         selectedLinks = []
    #         path = [current_node.id]
    #         while (not current_node == request[1]) and (hopCount < self.hopCountThreshold) and good_to_search:
                
    #             current_state, next_node_id = self.routingAgent.learn_and_predict_next_node(request, current_node , path)
    #             next_node = self.topo.nodes[next_node_id]
    #             ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
    #             key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

    #             if not len(ent_links):
    #                 good_to_search = False
    #                 # print((src.id,dst.id) , '=FAILED findDQRLPath = not len(ent_links)')

    #             else:
    #                 ent_links = [ent_links[0]]
    #                 for link in ent_links:
    #                     link.taken = True
    #                 selectedLinks.extend(ent_links)
                    

    #             # print(current_node.id , next_node_id , len(ent_links))


    #             if current_node == next_node:
    #                 good_to_search = False
    #                 # print((src.id,dst.id) , '=FAILED findDQRLPath = current_node == next_node')

                    
    #             if next_node.id in path:
    #                 good_to_search = False  
    #                 # print((src.id,dst.id) , '=FAILED findDQRLPath = loop')
                                 
                    
    #             if prev_node is None:
    #                 prev_links = ent_links
    #                 usedLinks.extend(prev_links)
                

    #             selectedNodes.append(next_node)
    #             selectedEdges.append((current_node, next_node))
    #             path.append(next_node.id)
                
    #             if len(prev_links) and next_node == request[1] and good_to_search:
    #                 success = True
    #                 good_to_search = False
                

                    
    #             prev_node = current_node
    #             current_node = next_node
    #             hopCount += 1
    #             # if hopCount >= self.hopCountThreshold:
    #                 # print((src.id,dst.id) , '=FAILED findDQRLPath = hopCount >= self.hopCountThreshold')

    #         for link in selectedLinks:
    #             link.taken = False
    #         # if success:
    #         #     print('======@@@@===== findDQRLPath len ' , len(selectedNodes) , 'for ' , (src.id , dst.id))

    #             return selectedNodes
    #         else:
    #             return []

    def extraRoute(self):
        T = []
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        pathlen = 0

        for request in  self.requests:
            T.append(request)
        # random.shuffle(T)
        T2 = []
        for req in T:
            T2.append(req)
        
        fp = 0

        for request in T:
        # while len(T2):
        #     tl = 999
        #     request = None
        #     for r in  T2:
        #         targetPath = self.findPathForDQRL((r[0],r[1]))
        #         if len(targetPath) < tl:
        #             request = r
        #             break
        #     T2.remove(request)


            usedLinks = []

            (src,dst) = (request[0] , request[1])
            if 'greedy_only' in self.name:
                targetPath = self.findPathForDQRL((src,dst))


            if not len(targetPath):
                continue
            fp += 1
            # print('********* path for ' , (src.id,dst.id) , ':: ' , len(targetPath) , ':: ' , ': ' , [n.id for n in targetPath])


            
            # print('========ent_links=======')
            # print((src.id, dst.id))

            current_node = src
            i =1
            prev_node = None
            prev_links = []
            hopCount = 0
            success = False
            good_to_search = True
            width = 0
            selectedNodes = []
            selectedEdges = []
            path = [current_node.id]


      
            while (not current_node == dst) and (hopCount < self.hopCountThreshold) and good_to_search:
                
                # current_state, next_node_id = self.routingAgent.learn_and_predict_next_node(request, current_node , path)
                next_node = targetPath[i]
                i+=1
                next_node_id = next_node.id
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped())]
                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                if not len(ent_links):
                    good_to_search = False
                    # print((src.id,dst.id) , '=FAILED extra route = not len(ent_links)')

                else:
                    ent_links = [ent_links[0]]

                # print(current_node.id , next_node_id , len(ent_links))


                if current_node == next_node:
                    good_to_search = False
                    # print((src.id,dst.id) , '=FAILED extra route = current_node == next_node')

                    
                if next_node.id in path:
                    good_to_search = False
                    # print((src.id,dst.id) , '=FAILED extra route = loop')

                
                # if good_to_search:
                #     # dist = current_state[self.routingAgent.env.SIZE *2 +2][next_node_id] +1
                #     dist = current_state[self.routingAgent.env.SIZE  +2][next_node_id] +1
                    # if not dist:
                    #     dist = 1

                    # try:
                    #     self.topo.reward_routing[key] += self.topo.positive_reward / dist
                    # except:
                    #     self.topo.reward_routing[key] = self.topo.positive_reward / dist
                    
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
                            # print((src.id,dst.id) , '=FAILED extra route = swap fails')

                            
                else:
                    prev_links = ent_links
                    usedLinks.extend(prev_links)
                

                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))
                path.append(next_node.id)
                # self.routingAgent.update_action( request , current_node.id,  next_node_id  , current_state , path , False)
                
                if len(prev_links) and next_node == request[1] and good_to_search:
                    success = True
                    good_to_search = False
                

                    
                prev_node = current_node
                current_node = next_node
                hopCount += 1
                # if hopCount >= self.hopCountThreshold:
                    # print((src.id,dst.id) , '=FAILED extra route = hopCount >= self.hopCountThreshold')

            
            if success:
                pathlen += len(targetPath)

                successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst)

                for path in successPath:
                    for node, link in path:
                        if link is not None:
                            link.used = True
                            edge = self.topo.linktoEdgeSorted(link)

                            self.topo.reward_ent[edge] =(self.topo.reward_ent[edge] + self.topo.positive_reward) if edge in self.topo.reward_ent else self.topo.positive_reward
                        # if node is not None:
                        #     try:
                        #         self.topo.reward_routing[(request , node)] += self.topo.positive_reward
                        #     except:
                        #         self.topo.reward_routing[(request , node)] = self.topo.positive_reward
                        #                     self.topo.reward_ent[edge] = self.topo.negative_reward
                    break

                # for (current_node, next_node) in selectedEdges:
                #     key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                #     # print(key)
                #     try:
                #         self.topo.reward_routing[key] += self.topo.positive_reward*2
                #     except:
                #         self.topo.reward_routing[key] = self.topo.positive_reward*2
                # print("!!!!!!!success!!!!!!!")

                # print([(r[0].id, r[1].id) for r in self.requests])

                # self.requests.remove(request)
                for req in self.requests:
                    src = req[0]
                    dst = req[1]
                    if (src, dst) == (request[0], request[1]):
                        # print('[REPS] finish time:', self.timeSlot - request[2])
                        self.requests.remove(req)
                        break

                successReq += 1
                totalEntanglement += len(successPath)
            else:
                for link in usedLinks:
                        edge = self.topo.linktoEdgeSorted(link)
                        try:
                            self.topo.reward_ent[edge] += self.topo.negative_reward
                        except:
                            self.topo.reward_ent[edge] = self.topo.negative_reward
                # for (current_node, next_node) in selectedEdges:
                #     key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)

                #     # try:
                #     #     self.topo.reward_routing[key] += self.topo.negative_reward*20
                #     # except:
                #     #     self.topo.reward_routing[key] = self.topo.negative_reward*20

                #     try:
                #         self.topo.reward_routing[key] += 0
                #     except:
                #         self.topo.reward_routing[key] = 0
            
            for link in usedLinks:
                link.clearPhase4Swap()

        print('===== paths found =======================================' , fp)

        if fp > 0:
            self.result.pathlen += pathlen
        self.result.totalPath += fp
        
        self.result.usedLinks += len(usedLinks)

        return successReq , totalEntanglement


    def findPathForDQRL(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        if self.DijkstraForDQRL(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]
            path = path[::-1]
            return path
        else:
            return []
    
    def DijkstraForDQRL(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if 0 < self.edgeSuccessfulEntangle(node1, node2):
                    adjcentList[node1].add(node2)
        
        # for node in adjcentList:
        #     print([n.id for n in adjcentList[node]])
        
        distance = {node : math.inf for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        # print(src.id)
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
                newDistance = distance[u] + self.weightofLink(u,next)
                if distance[next] > newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((distance[next], next.id))

        return False
    def weightofLink(self , u , v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken:
                capacity += 1

        return 1/capacity
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
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
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
          
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken:
                capacity += 1
        # print(capacity)
        return capacity
    def edgeSuccessfulEntangleForELS(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken and not link.considered:
                capacity += 1
        # print(capacity)
        return capacity

    def LP2(self):
        # print('[REPS] LP2 start')
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow = [1 for i in range(numOfSDpairs)]
        if len(numOfFlow):
            maxK = max(numOfFlow)
        else:
            maxK = 0
        # maxK = 1
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
                            f[i][k][u][v] = m.addVar(lb = 0, vtype = gp.GRB.INTEGER, name = "f[%d][%d][%d][%d]" % (i, k, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxK
            for k in range(maxK):
                if k < numOfFlow[i]:
                    t[i][k] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.INTEGER, name = "t[%d][%d]" % (i, k))
                else:
                    t[i][k] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.INTEGER, name = "t[%d][%d]" % (i, k))

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
                            # if v not in [s, d] and (u,v) in edgeIndices:
                            if (u,v) in edgeIndices or (v,u) in edgeIndices:
                                edgeUV.append(v)
                            # print('edgeUV ' , u , edgeUV)
                        m.addConstr(gp.quicksum(f[i][k][u][v] for v in edgeUV) - gp.quicksum(f[i][k][v][u] for v in edgeUV) == 0)

        
        for (u, v) in edgeIndices:
            capacity = self.edgeSuccessfulEntangle(self.topo.nodes[u], self.topo.nodes[v])
            # print('capcity ' ,(u , v),  capacity)
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

                    # print('self.fki_LP[SDpair][k][(u, v)]' , SDpair[0].id , SDpair[1].id , k , (u.id, v.id) ,  self.fki_LP[SDpair][k][(u, v)])
                    # print('self.fki_LP[SDpair][k][(v, u)]' , SDpair[0].id , SDpair[1].id , k , (v.id, u.id) ,  self.fki_LP[SDpair][k][(v, u)])


                # for (u, v) in notEdge:
                #     u = self.topo.nodes[u]
                #     v = self.topo.nodes[v]
                #     self.fki_LP[SDpair][k][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, k])
                self.tki_LP[SDpair][k] = m.getVarByName(varName).x
                # print('self.tki_LP[SDpair][k]' , SDpair[0].id , SDpair[1].id , k , self.tki_LP[SDpair][k])
        # print('[REPS] LP2 end')

    def EPS(self):
        self.LP2()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : 1 for SDpair in self.srcDstPairs}
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
                # print('self.tki_LP[SDpair][k]' , SDpair[0].id ,SDpair[1].id  , k , self.tki_LP[SDpair][k])
                self.tki[SDpair][k] = self.tki_LP[SDpair][k]

                if not self.tki[SDpair][k]:
                    continue
                paths = self.findPathsForEPS(SDpair, k)

                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0
                    
                for path in paths:
                    width = path[-1]
                    # select = (width / self.tki_LP[SDpair][k]) >= random.random()
                    select = 1
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

    def ELS(self , Pi):
        Ci = self.pathForELS
        self.y = {(u, v) : 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        needLink = {}
        nextLink = {node : [] for node in self.topo.nodes}
        T = [SDpair for SDpair in self.srcDstPairs]
        for sd in Pi:
            if len(Pi[sd]) and sd in T:
                T.remove(sd)
        output = []
        while len(T) > 0 :
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangleForELS(node, next):
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
            if len(targetPath) ==2:
                for link in targetPath[0].links:
                    if link.contains(targetPath[1]) and link.entangled and link.notSwapped() and not link.taken and not link.considered:
                        targetLink1 = link
                        break
                targetLink1.considered = True
            for nodeIndex in range(1, len(targetPath) - 1):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                for link in node.links:
                    if link.contains(next) and link.entangled and link.notSwapped() and not link.taken and not link.considered:
                        targetLink1 = link
                    
                    if link.contains(prev) and link.entangled and link.notSwapped() and not link.taken and not link.considered:
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1

                nextLink[node].append(targetLink1)
                targetLink1.considered = True
                if nodeIndex == 1:
                    targetLink2.considered = True
                needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))

            T.remove(i)
        print('** before graph ' , len(output))
        print([(path[0].id , path[-1].id) for path in output])

        print('** after graph ' , len(output))





        return len(output)

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
                # newDistance = distance[u] + self.weightOfNode[next]
                newDistance = distance[u] + self.weightofLink(u,next)
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