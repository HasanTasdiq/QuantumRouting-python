from dataclasses import dataclass
from re import A
import random
import math
import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import PickedPath
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link

class MyAlgorithm(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.name = "My"
        self.r = 40 # 暫存回合
        self.givenShortestPath = {} # {(src, dst): path, ...}
        self.socialRelationship = {}    # {id : [Node, ...], ...}
        self.requestState = {}  # {(src, dst, timeslot) : [state, Node(k), path length, path, take]}

    def establishShortestPath(self):        
        for n1 in self.topo.nodes:
            for n2 in self.topo.nodes:
                if n1 != n2:
                    self.givenShortestPath[(n1, n2)] = self.topo.shortestPath(n1, n2, 'notHop')[1] 
    
    def genSocialRelationalship(self):
        for n in self.topo.nodes:
            cnt = int(len(n.neighbors) * 0.5)    # 50%
            sample = random.sample(n.neighbors, cnt)
            self.socialRelationship[n] = sample

    # p1
    def descideSegmentation(self):
        for req in self.srcDstPairs:
            src, dst = req[0], req[1]
            self.requestState[(src, dst, self.timeSlot)] = [0, None, len(self.givenShortestPath[(src, dst)]), self.givenShortestPath[(src, dst)], False]
            D_sd = self.topo.distance(src.loc, dst.loc)
            P_sd = math.exp(-self.topo.alpha * D_sd)
            minNum = 1 / P_sd
            

            for k in self.socialRelationship[src]:
                D_sk = self.topo.distance(src.loc, k.loc)
                D_kd = self.topo.distance(k.loc, dst.loc)
                P_sk = math.exp(-self.topo.alpha * D_sk)
                P_kd = math.exp(-self.topo.alpha * D_kd)
                curMin = math.ceil(1 / (self.r * P_kd)) * (1 / P_sk) + (1 / P_kd)

                if minNum > curMin:    # 分2段 取k中間  
                    minNum = curMin
                    self.requestState[(src, dst, self.timeSlot)] = [1, k, len(self.givenShortestPath[(src, k)]), self.givenShortestPath[(src, k)], False]

    def prepare(self):
        self.givenShortestPath.clear()
        self.socialRelationship.clear()
        self.establishShortestPath()
        self.genSocialRelationalship()

    # p2 第2次篩選
    def p2Extra(self):
        for req in self.requestState:
            state, take = self.requestState[req][0], self.requestState[req][3]
            if state == 0: 
                src, dst, time = req[0], req[1], req[2]
            else:
                src, dst, time = req[0], self.requestState[req][1], req[2]

            if not take:
                if src.remainingQubits < 1:
                    continue
                p = []
                p.append(src)
                
                # Find a shortest path by greedy min hop  
                while True:
                    last = p[-1]
                    if last == dst:
                        break

                    # Select avaliable neighbors of last(local)
                    selectedNeighbors = []    # type Node
                    selectedNeighbors.clear()
                    for neighbor in last.neighbors:
                        if neighbor.remainingQubits > 2 or neighbor == dst and neighbor.remainingQubits > 1:
                            for link in neighbor.links:
                                if link.contains(last) and (not link.assigned):
                                    # print('select neighbor:', neighbor.id)
                                    selectedNeighbors.append(neighbor)
                                    break

                    # Choose the neighbor with smallest number of hop from it to dst
                    next = self.topo.sentinal
                    hopsCurMinNum = sys.maxsize
                    for selectedNeighbor in selectedNeighbors:
                        hopsNum = self.topo.hopsAway(selectedNeighbor, dst, 'Hop')      
                        if hopsCurMinNum > hopsNum:
                            hopsCurMinNum = hopsNum
                            next = selectedNeighbor

                    # If have cycle, break
                    if next == topo.sentinal or next in p:
                        break 
                    p.append(next)
                # while end

                if p[-1] != dst:
                    continue
                
                # Caculate width for p
                width = topo.widthPhase2(p)
                
                if width == 0:
                    continue
                
                # Assign Qubits for links in path 
                for i in range(0, width):
                    for s in range(0, len(p) - 1):
                        n1 = p[s]
                        n2 = p[s+1]
                        for link in n1.links:
                            if link.contains(n2) and (not link.assigned):
                                link.assignQubits()
                                break 

                if state == 1:
                    dst.assignIntermediate()
                
                self.requestState[req][3] = p
                self.requestState[req][4] = True

    # p1 & p2    
    def p2(self):

        # p1
        self.descideSegmentation()

        # 根據path長度排序
        sorted(self.requestState.items(), key=lambda x: x[1][2])
        self.requestState = dict(self.requestState)

        # p2 第1次篩選
        for req in self.requestState:
            state = self.requestState[req][0]
            if state == 0: 
                src, dst, time = req[0], req[1], req[2]
            else:
                src, dst, time = req[0], self.requestState[req][1], req[2]

            # 檢查path資源
            path = self.givenShortestPath[(src, dst)]
            unavaliable = False
            for n in path:
                if (state == 0 and (n == src or n == dst) and n.remainingQubits < 1) or \
                    (state == 1 and n == dst and n.remainingQubits < 2) or \
                    (state == 1 and n == src and n.remainingQubits < 1) or \
                    (state == 2 and n == src and n.remainingQubits < 1) or \
                    (state == 2 and n == dst and n.remainingQubits < 1) or \
                    ((n != src and n != dst) and n.remainingQubits < 2):             
                    unavaliable = True

            # 檢查link資源
            for s in range(0, len(path) - 1):
                n1 = path[s]
                n2 = path[s+1]
                pick = False
                for link in n1.links:
                    if link.contains(n2) and (not link.assigned):
                        pick = True

                if not pick:
                   unavaliable = True  
            
            # 資源不夠 先跳過
            if unavaliable:
                continue

            # 分配資源給path
            for s in range(0, len(path) - 1):
                n1 = path[s]
                n2 = path[s+1]
                for link in n1.links:
                    if link.contains(n2) and (not link.assigned):
                        link.assignQubits()
                        break 

            # 有分段 另外分配資源給中繼點
            if state == 1:
                dst.assignIntermediate()
            
            # take這個request
            self.requestState[req][3] = path
            self.requestState[req][4] = True
        

        # p2 第2次篩選
        self.p2Extra()


if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.01, 6)
    s = MyAlgorithm(topo)
    s.work([(topo.nodes[3],topo.nodes[99])], 0)