from multiprocessing import Lock, Manager
import pickle
import sys
import math
import random
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 

from numpy import log as ln
from random import sample
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from concurrent.futures import ProcessPoolExecutor
from topo.mp_helper import executor as executor2, route_schedule_single2 , route_schedule_single, qManager
from multiprocessing.managers import BaseManager


# executor2 = ProcessPoolExecutor(max_workers=8)  # Create at the top level




# ctx._force_start_method('spawn')

sys.path.insert(0, "../../rl")
max_workers = os.cpu_count()

from DQRLAgentDist import DQRLAgentDist


class QuRA_DQRL_DIST(AlgorithmBase):
    def __init__(self, topo,param=None, name=''):
        super().__init__(topo)
        self.name = name
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        # self.entAgent = DQNAgentDistEnt(self, 0)
        self.routingAgent = DQRLAgentDist(self , 0)
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        self.hopCountThreshold = 25
        self.requestState = []
        self.optPaths = {}
        if 'greedy_only' not in self.name:
            self.routingAgent.initiate()
        # self.pool = None
        self.w1 = 1
        self.w2 = 1 - self.w1
        self.maxTry = 2
        self.executor = None
        self.tst = [] 



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
            # self.PFT() # compute (self.ti, self.fi)
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
        p_time = 0
        global executor2

        # self.prep4()

        if len(self.srcDstPairs) > 0:
            # self.EPS()
            # self.ELS()
            t = time.time()

            if 'greedy_only' in  self.name:
                self.route_seq()
                p_time += time.time()-t
            else:
                # self.route()
                # with ThreadPoolExecutor(max_workers=8) as executor:
                # with ProcessPoolExecutor(max_workers=8) as executor:
                # if not executor2:
                #     executor2 = ProcessPoolExecutor(max_workers=8)
                # with Manager() as manager:
                #     shared_nodes = manager.dict({node.id: {"remainingQubits": node.remainingQubits} for node in self.topo.nodes})
  
                args = [( reqState) for reqState in self.requestState]
                # self.topo.tst = Manager().list()
                # for _ in range(10):
                #     print('going to map route_schedule_single with args2:' , len(args), len(args[0]))
                #     # route_schedule_single2(args[0])
                
                results = list(executor2.map(self.route_schedule_single, args))
                print('results ' , results, sum([r for r in results]))
            
                successReq = sum([r for r in results])
                self.result.successfulRequestPerRound.append(successReq)
                self.result.entanglementPerRound.append(sum([r for r in results]))
                self.result.fidelityPerRound.append(0)

                self.result.successfulRequest += successReq
                # p_time += self.route_schedule_seq()
            # self.route_seq()
            # self.route_schedule_seq()
            print('time for route schedule ======== ' , time.time() - t)
        t = time.time()
        self.filterReqeuest()

        # print('[REPS] p4 end') 
        self.printResult()
        # self.entAgent.update_reward()
        reward = 0
        if not 'greedy_only' in self.name:
            if self.timeSlot <= 500000:
                t = time.time()
                
                reward = self.routingAgent.update_reward(self.result.successfulRequestPerRound[-1], self.result.fidelityPerRound[-1])
                reward = 0
                print('time for update_reward ======== ' , time.time() - t)

            a = 10
        self.result.rewardPerRound.append(reward)
        p_time += time.time()-t
        self.result.p_time = p_time
        print('self.w1 w2 ' , self.w1 , self.w2)

        return self.result
    def get_action_ILP(self):
        req_index = [req[4] for req in self.requestState if not req[[5]]]
        req = self.requestState[random.choice(req_index)]
        next_node  = self.Pi[req][0][1]
        self.Pi[req][0].pop(0)
        
    

    def getConflicts(self, current_node , next_node , selectedEdgesDict):
        conflicts = []
        for req in selectedEdgesDict:
            edges = selectedEdgesDict[req]
            if (current_node, next_node) in edges:
                conflicts.append((req , current_node, next_node))
            elif (next_node, current_node) in edges:
                conflicts.append((req , next_node, current_node))
        
        return conflicts



    def route_schedule_single(self ,  reqState):
        """
        Serve only one request (reqState) using the routing agent.
        reqState: [src, dst, current_node, path, index, checked]
        """
        # topo = pickle.loads(serialized_topo)
        # nodes = pickle.loads(serialized_nodes)
        src, dst, current_node, path, index, checked = reqState
        selectedNodes = [src]
        selectedEdges = []
        selectedlinks = []
        usedLinks = []
        prev_links = None
        hopCount = 0
        success = False
        good_to_search = True
        failed_no_ent = False
        failed_loop = False
        failed_swap = False
        fail_hopcount = False
        path = list(path)
        numtry = 0
        maxTry = self.maxTry
        fidelity = 1

        while good_to_search and not success and numtry <= maxTry:
            # Get next action for this request

            with Lock():
                result = self.routingAgent.learn_and_predict_next_req_node_single(reqState)
                if result is None:
                    break
                current_state, req_id, next_node_id, q, mask, valid_actions = result
                next_node = self.topo.nodes[next_node_id]
                self.tst.append(os.getpid())
                print('process id:', os.getpid(), 'thread id:', threading.get_ident(), 'next_node_id:', next_node_id , next_node , set(self.tst))
                current_node_id = current_node.id
                # Find entangled links
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                print(f"Processing request {src.id} to {dst.id},current node ID: {current_node.id} next node ID: {next_node_id}", 'len ent_links:', len(ent_links) , 'path:', path)
                key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                
                if not ent_links:

                    numtry += 1
                    if numtry <= self.maxTry:
                        continue
                    else:
                        good_to_search = False
                        failed_no_ent = True
                else:
                    ent_links = [ent_links[0]]
                    ent_links[0].taken = True
                    selectedlinks.append(ent_links[0])
                    prev_links = [ent_links[0]]
                    numtry = 0

                    # Fidelity check
                    fidelity = self.fidelityAfterSwap(fidelity, ent_links[0].fidelity)
                    if fidelity < self.topo.fidelity_threshold:
                        numtry += 1
                        if numtry <= self.maxTry:
                                continue
                        else:
                            numtry = 0
                            good_to_search = False


                # Loop check
                if next_node == current_node or next_node.id in path:
                    good_to_search = False
                    failed_loop = True
                    # break

                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))
                # usedLinks.extend(prev_links)
                path.append(next_node.id)
                req_done = (not good_to_search) or success

                reqState = (src,dst,next_node,tuple(path),index,req_done)
                self.requestState[index] = reqState
                # print('Updated request state:', reqState)
                # Success check
                if next_node == dst and good_to_search:
                    success = True
                    good_to_search = False

                # Prepare for next hop
                current_node = next_node
                swappSuccess = True
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
                            swappSuccess = False
                            print('================failed swap==================')
                            break
                


                if success and swappSuccess:
                    print('going to find path for:', (src.id , dst.id ))
                    t2 = time.time()
                    for req in self.requests:
                            # src = req[0]
                            # dst = req[1]
                        if (src, dst) == (req[0], req[1]):
                                # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(req)
                            break

                    # successReq += 1
                    # totalEntanglement += 1


                reward = -1

                if req_done:
                    if success:
                            # print("====success====" , src.id , dst.id , [n for n in path])
                            # print('shortest path ----- ' , [n.id for n in targetPath])
                        reward = 10
                            # reward = 1

                        total_fidelity += fidelity
                    else:
                        for link in selectedlinks:
                            link.taken = False
                        print("!!!!!!!=fail=!!!!!!!" , src.id , dst.id , [n for n in path] , 'threading.get_ident():', threading.get_ident())
                            # print('shortest path ----- ' , [n.id for n in targetPath])
                        
                        print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                        reward = -10


                    # print('lenT ' , len(T))

                # key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(prev_node.id) + '_' + str(next_node.id)

                    # reward = -self.topo.numOfRequestPerRound

                try:
                    self.topo.reward_routing[key] += reward
                except:
                    self.topo.reward_routing[key] = reward    

                for link in usedLinks:
                    link.clearPhase4Swap()
                
                T = [r for r in self.requestState if not r[5]]
                done_episode = (not good_to_search or success) and (len(T)==1)
            
            with Lock():
                self.routingAgent.update_action( reqState ,current_node_id,  next_node_id  , current_state  , done_episode)
            

        return success and swappSuccess

    def route_schedule_seq(self):
        successReq = 0
        totalEntanglement = 0
        usedLinks = []
        conflicts = []
        p_time = 0
        t_1 = time.time()
        total_action = 0
        total_fidelity = 0
  

        T = []
        for request in  self.requests:
            T.append(request)
        
        selectedNodesDict = {}
        selectedEdgesDict = {}
        prevlinksDict = {}
        usedLinksDict = {}
        selectedlinksDict = {}
        tryDict = {}
        fidelityDict = {}

        for reqState in self.requestState:
            src, dst = reqState[0] , reqState[1]
            selectedNodesDict[(src, dst)] = [src]
            selectedEdgesDict[(src, dst)] = []
            selectedlinksDict[(src,dst)] = []
            usedLinksDict[(src, dst)] = []
            prevlinksDict[(src,dst)] = None
            tryDict[(src,dst)] = 0
            fidelityDict[(src,dst)] = 1

        conflicts = []

        # print(self.name , ('greedy_only' in  self.name))
        # print([(r[0].id , r[1].id) for r in self.srcDstPairs])
        print([(r[0].id , r[1].id) for r in self.requests])
        # if not (self.param is not None and 'greedy_only' in self.param):

        # if True:
        while len(T):
            t = time.time()
            reqState_actions = self.routingAgent.learn_and_predict_next_req_node_all()
            # print('learn and predict time ' , time.time()-t)
            actions = []
            for (current_state , req_id , next_node_id , q , mask , valid_actions) in reqState_actions:
                action = next_node_id
                if next_node_id < 0:
                    continue
                p_time2 = 0


                total_action += 1
                
                t1 = time.time()
                # current_state , action , p_time2 = self.routingAgent.learn_and_predict_next_req_node()
                # print('learn_and_predict_next_req_node time: ' , time.time() - t1)

                p_time+= p_time2
                t2 = time.time()


                # print('req iddddd ' , req_id , next_node_id)
                # print([(req[0].id , req[1].id , req[2].id, req[4]) for req in self.requestState])
                # req_id , next_node_id = self.routingAgent.decode_schdeule_route_action(action)

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
                numtry = tryDict[(src,dst)]
                path = list(path)
                failed_no_ent = False
                failed_loop = False
                failed_swap = False
                fail_hopcount = False
                # targetPath = self.findPathForDQRL((current_node,dst))
                # if not len(targetPath):
                #     good_to_search = False
                # targetPath = []
                skipRequest = False
                # if not len(targetPath):
                #     continue
                # targetPath = []

                    
                ent_links = []
                # print('*************************' , action , 'valid actions: ' , valid_actions)
                for i in range(min(3,len(valid_actions))):
                    next_node_id = valid_actions[i]
                    next_node = self.topo.nodes[next_node_id]
                    ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                    ent_links_count = len(ent_links)
                    action = next_node_id

                    if ent_links_count:
                        # print('ent links count ' , ent_links_count)
                        break   
                key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                
                # next_node = self.topo.nodes[next_node_id]
                # ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                # ent_links_count = len(ent_links)

                # print(src.id , dst.id , action , path)
                if not len(ent_links):
                    good_to_search = False
                    failed_no_ent = True
                    numtry += 1
                    tryDict[(src,dst)] = numtry
                    if numtry <= self.maxTry:
                        continue
                    # conflicts_ = self.getConflicts(current_node , next_node , selectedEdgesDict)
                    # if len(conflicts_):
                    #     failed_conflict = True
                    # conflicts.extend(conflicts_)
                        # print((src.id,dst.id) , '=FAILED= no ent links')
                else:
                    ent_links = [ent_links[0]]
                    for link in ent_links:
                        link.taken = True
                    numtry = 0
                    tryDict[(src,dst)] = numtry

                    # print(current_node.id , next_node_id , len(ent_links))
                fid = fidelityDict[(src,dst)]
                if good_to_search:
                    fid2 = self.fidelityAfterSwap(fid , ent_links[0].fidelity)
                    if fid2 < self.topo.fidelity_threshold:
                        numtry += 1
                        tryDict[(src,dst)] = numtry 
                        if numtry <= self.maxTry:
                            continue
                        else:
                            numtry = 0
                            tryDict[(src,dst)] = numtry

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

                fidelity = 0

                    
                if len(prev_links) and next_node == request[1] and good_to_search:
                    success = True
                    good_to_search = False

                if len(selectedlinks):
                    fidelity = selectedlinks[0].fidelity
                    for link in selectedlinks[1:]:
                        fidelity = self.fidelityAfterSwap(fidelity , link.fidelity)
                    if fidelity < self.topo.fidelity_threshold:
                        success = False
                        req_done = True
                done_episode = (not good_to_search or success) and (len(T)==1)
                t3 = time.time()
                actions.append(( reqState ,current_node.id,  action  , current_state  , done_episode))
                # print('update action time ' , time.time()-t3)
                        
                prev_node = current_node
                current_node = next_node
                req_done = (not good_to_search) or success
                # print('good_to_search ' , good_to_search , 'req_done ' , req_done)
                reqState = (src,dst,current_node,tuple(path),index,req_done)
                self.requestState[index] = reqState
                swappSuccess = True


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
                            swappSuccess = False
                            print('================failed swap==================')
                            break
             


                if success and swappSuccess:
                    print('going to find path for:', (src.id , dst.id ))
                    t2 = time.time()
                    for req in self.requests:
                        # src = req[0]
                        # dst = req[1]
                        if (src, dst) == (req[0], req[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(req)
                            break

                    successReq += 1
                    totalEntanglement += 1

                    


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
                        # print("====success====" , src.id , dst.id , [n for n in path])
                        # print('shortest path ----- ' , [n.id for n in targetPath])
                        reward = 10
                        # reward = 1

                        total_fidelity += fidelity
                    else:
                        for link in selectedlinks:
                            link.taken = False
                        print("!!!!!!!=fail=!!!!!!!" , src.id , dst.id , [n for n in path])
                        # print('shortest path ----- ' , [n.id for n in targetPath])
                        print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                        reward = -10


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
                p_time += time.time()-t2

            for (reqState , current_node_id , action  , current_state  , done_episode) in actions:
                self.routingAgent.update_action( reqState ,current_node_id,  action  , current_state  , done_episode)

        t2 = time.time()
        self.result.usedLinks += len(usedLinks)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request before extra:', successReq)
        print('[' , self.name, '] :' , self.timeSlot, ' =================conflicts :', len(conflicts))


        extra_successReq , extra_totalEntanglement = 0 , 0
        if 'greedy_only' in self.name or 'bruteforce' in self.name:
            extra_successReq , extra_totalEntanglement = self.extraRoute()

        totalEntanglement += extra_totalEntanglement
        successReq += extra_successReq
        try:
            avgFidelity = total_fidelity/successReq
        except:
            avgFidelity = 0    
        self.result.entanglementPerRound.append(totalEntanglement)
        self.result.successfulRequestPerRound.append(successReq)
        self.result.fidelityPerRound.append(avgFidelity)

        self.result.successfulRequest += successReq
        # self.result.fidelity += avgFidelity
        
        entSum = sum(self.result.entanglementPerRound)
        self.filterReqeuest()
        print(self.name , '######+++++++========= total ent: '  , 'till time:' , self.timeSlot , ':=' , entSum)
        print('[' , self.name, '] :' , self.timeSlot, ' current successful request  after  extra:', successReq)
        print('[' , self.name, '] :' , self.timeSlot, ' total time for route ' , (time.time()-t_1) , ' action ' , total_action , 'average time ' , (time.time()-t_1)/(total_action if total_action else 1))
    
        p_time += time.time()-t2
        return p_time

        

    


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
        # w1 = .5
        # w2 = .5

        w1 = self.w1
        w2 = self.w2
        if u == v:
            return 0
        capacity = 0
        fidelity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken:
                capacity += 1
                fidelity = max(fidelity , link.fidelity)

        return w1/capacity + w2 * (-math.log(fidelity))
        # return w1/capacity + w2/fidelity
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