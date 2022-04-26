from dataclasses import dataclass
from re import A
import random
from random import sample
import math
import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import PickedPath
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link

@dataclass
class RequestInfo:
    state: int          # 0 代表src直接送到dst, 1 代表src送到K, 2 代表k送到dst
    intermediate: Node  # 中繼點 k
    pathlen: int        
    pathseg1: list
    pathseg2: list 
    taken : bool        # 是否可處理這個req (已預定資源)
    savetime : int      # req存在中繼點k 還沒送出去經過的時間
    linkseg1 : list     # seg1 用過的 links

class MyAlgorithm(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.name = "My"
        self.r = 40                     # 暫存回合
        self.givenShortestPath = {}     # {(src, dst): path, ...} path表
        self.socialRelationship = {}    # {Node : [Node, ...], ...} social表
        self.requestState = {}          # {(src, dst, timeslot) : RequestInfo}  這回合要做的request
        self.totalTime = 0
 
    def establishShortestPath(self):        
        for n1 in self.topo.nodes:
            for n2 in self.topo.nodes:
                if n1 != n2:
                    self.givenShortestPath[(n1, n2)] = self.topo.shortestPath(n1, n2, 'Hop')[1] 

    def Pr(self, path):
        P = 1
        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            d = self.topo.distance(n1.loc, n2.loc)
            p = math.exp(-self.topo.alpha * d)
            P *= p

        return P
      
    def expectedRound(self, p1, p2):
        aa = 0 
        a = 0
        b = 0
        i = 0 
        while(1):
            k = i-math.ceil(i/(self.r+1))
            for j in range(1,k+1):
                b += math.factorial(i-j-1)//math.factorial(math.ceil(j/self.r)-1)//math.factorial(i-j-math.ceil(j/self.r))*pow(p1,math.ceil(j/self.r))*pow((1-p1),i-j-math.ceil(j/self.r))*p2*pow((1-p2),j-1)
            aa += a
            a += i*b

            if aa !=0 and a/aa <= 0.005 :
                break
            b = 0
            i += 1
        return a

    def genSocialRelationalship(self):
        for i in range(0, len(self.topo.nodes)):
            for j in range(i, len(self.topo.nodes)):
                n1 = self.topo.nodes[i]
                n2 = self.topo.nodes[j]
                if i == j:
                    continue
                p = random.random() + 0.05
                if p >= 0.5:
                    self.socialRelationship[n1].append(n2)
                    self.socialRelationship[n2].append(n1)

    # p1
    def descideSegmentation(self):
        nodeRemainingQubits = {node: node.remainingQubits for node in self.topo.nodes}

        # 在前面回合 要拆成2段 但送失敗(還在src) 先把預定的bit扣掉 
        for req in self.requestState:
            if self.requestState[req].state == 1:
                k = self.requestState[req].intermediate
                nodeRemainingQubits[k] -= 1

        # 針對新的req 決定要不要拆
        for req in self.srcDstPairs:
            src, dst = req[0], req[1]
            path_sd = self.givenShortestPath[(src, dst)]
            self.requestState[(src, dst, self.timeSlot)] = RequestInfo(0, None, len(path_sd), path_sd, None, False, 0, None)
            P_sd = self.Pr(self.givenShortestPath[(src, dst)])
            minNum = 1 / P_sd
            # print('minNum:', minNum)
            
            for k in self.socialRelationship[src]:
                if nodeRemainingQubits[k] <= 1 or k == dst:
                    continue
                path_sk = self.givenShortestPath[(src, k)]
                path_kd = self.givenShortestPath[(k, dst)]
                P_sk = self.Pr(path_sk)
                P_kd = self.Pr(path_kd)
                curMin = self.expectedRound(P_sk, P_kd)
                # print('curMin:', curMin)
                if minNum > curMin:    # 分2段 取k中間  
                    minNum = curMin
                    self.requestState[(src, dst, self.timeSlot)] = RequestInfo(1, k, len(path_sk), path_sk, path_kd, False, 0, None)

            # 模擬用掉這個k的一個Qubits，非真正用掉
            k = self.requestState[(src, dst, self.timeSlot)].intermediate
            if k == None: continue
            nodeRemainingQubits[k] -= 1
        

    def prepare(self):
        self.givenShortestPath.clear()
        self.socialRelationship.clear()
        self.socialRelationship = {node: [] for node in self.topo.nodes}
        self.establishShortestPath()
        self.genSocialRelationalship()

    # p2 第2次篩選
    def p2Extra(self):
        for req in self.requestState:
            requestInfo = self.requestState[req]

            if requestInfo.state == 0: 
                src, dst = req[0], req[1]
            elif requestInfo.state == 1:
                src, dst = req[0], requestInfo.intermediate
            elif requestInfo.state == 2:
                src, dst = requestInfo.intermediate, req[1]

            if not requestInfo.taken:
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
                    next = self.topo.sentinel
                    hopsCurMinNum = sys.maxsize
                    for selectedNeighbor in selectedNeighbors:
                        hopsNum = self.topo.hopsAway(selectedNeighbor, dst, 'Hop')      
                        if hopsCurMinNum > hopsNum:
                            hopsCurMinNum = hopsNum
                            next = selectedNeighbor

                    # If have cycle, break
                    if next == self.topo.sentinel or next in p:
                        break 
                    p.append(next)
                # while end

                if p[-1] != dst:
                    continue
                
                # Caculate width for p
                width = self.topo.widthPhase2(p)
                
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

                if requestInfo.state == 1:
                    dst.assignIntermediate()
                
                if requestInfo.state == 2:
                    requestInfo.pathseg2 = p
                else:
                    requestInfo.pathseg1 = p
                requestInfo.taken= True

                print('P2Extra take')

    def resetFailedRequestFor01(self, requestInfo, usedLinks):                   # 第一段傳失敗
        # for link in usedLinks:
        #     link.clearPhase4Swap()
        
        requestInfo.taken = False
        for link in usedLinks:
            link.clearEntanglement()

    def resetFailedRequestFor2(self, requestInfo, usedLinks):       # 第二段傳失敗 且超時
        requestInfo.savetime = 0
        requestInfo.state = 1
        requestInfo.pathlen = len(requestInfo.pathseg1)
        requestInfo.intermediate.clearIntermediate()
        # requestInfo.taken = False # 這邊可能有問題 重新分配資源

        # 第二段的資源全部釋放
        for link in usedLinks:
            link.clearEntanglement()    
    
    def resetSucceedRequestFor1(self, requestInfo, usedLinks):      # 第一段傳成功
        requestInfo.state = 2
        requestInfo.pathlen = len(requestInfo.pathseg2)
        requestInfo.taken = False                           # 這邊可能有問題 重新分配資源
        requestInfo.linkseg1 = usedLinks                    # 紀錄seg1用了哪些link seg2成功要釋放資源

        # 第一段的資源還是預留的 只是清掉entangled跟swap
        for link in usedLinks:      
            link.clearPhase4Swap()

    def resetSucceedRequestFor2(self, requestInfo, usedLinks):      # 第二段傳成功 
        # 資源全部釋放
        requestInfo.intermediate.clearIntermediate()
        for link in usedLinks:
            link.clearEntanglement()
        for link in requestInfo.linkseg1: 
            link.clearEntanglement()

    # p1 & p2    
    def p2(self):

        # p1
        self.descideSegmentation()

        # 根據path長度排序 
        sorted(self.requestState.items(), key=lambda x: x[1].pathlen)
        self.requestState = dict(self.requestState)

        # p2 (1)
        for req in self.requestState:
            requestInfo = self.requestState[req]
            if requestInfo.taken == True:   # 被選過了 可能是上一輪失敗的 已經配好資源了
                continue

            if requestInfo.state == 0:      # 0
                src, dst = req[0], req[1]
            elif requestInfo.state == 1:    # 1
                src, dst = req[0], requestInfo.intermediate
            elif requestInfo.state == 2:    # 2
                src, dst = requestInfo.intermediate, req[1]

            # 檢查path qubit資源
            path = self.givenShortestPath[(src, dst)]
            unavaliable = False
            for n in path:
                if ((n == src or n == dst) and n.remainingQubits < 1) or \
                    (requestInfo.state == 1 and n == src and n.remainingQubits < 1) or \
                    (requestInfo.state == 1 and n == dst and n.remainingQubits < 2) or \
                    (requestInfo.state == 2 and n == src and n.remainingQubits < 1) or \
                    (requestInfo.state == 2 and n == dst and n.remainingQubits < 1) or \
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
                        continue

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
            if requestInfo.state == 1:
                dst.assignIntermediate()
            
            # take這個request
            if requestInfo.state == 2:
                requestInfo.pathseg2 = path
            else:
                requestInfo.pathseg1 = path
            requestInfo.taken= True
        
        # p2 (2) 對資源不夠的 另外找路徑 
        self.p2Extra()

  
    # p4 & p5
    def p4(self):
        
        sorted(self.requestState, key=lambda q: q[2])
        finishedRequest = []
        # p4
        for req in self.requestState:
            requestInfo = self.requestState[req]
            if not requestInfo.taken:
                continue
            
            print('-----------------')
            print('src:', req[0].id, 'dst:', req[1].id)
            
            # swap
            if requestInfo.state == 2:
                p = requestInfo.pathseg2
            else:
                p = requestInfo.pathseg1

            usedLinks = set()

            for i in range(1, len(p) - 1):
                prev = p[i-1]
                curr = p[i]
                next = p[i+1]
                prevLinks = []
                nextLinks = []
                
                for link in curr.links:
                    if link.entangled and (link.n1 == prev and not link.s2 or link.n2 == prev and not link.s1):
                        prevLinks.append(link)
                        break

                for link in curr.links:
                    if link.entangled and (link.n1 == next and not link.s2 or link.n2 == next and not link.s1):
                        nextLinks.append(link)
                        break

                for (l1, l2) in zip(prevLinks, nextLinks):
                    usedLinks.add(l1)
                    usedLinks.add(l2)
                    curr.attemptSwapping(l1, l2)
            
            if len(p) == 2:
                prev = p[0]
                curr = p[1]
                for link in prev.links:
                    if link.entangled and (link.n1 == prev and not link.s2 or link.n2 == prev and not link.s1):
                        usedLinks.add(link)
                        break
         
            # p5
            success = len(self.topo.getEstablishedEntanglements(p[0], p[-1]))

            print('-----------------')
            print('success:', success)
            print('state:'  , requestInfo.state)
            pp = self.givenShortestPath[(req[0],req[1])]
            print('original path:'   , [x.id for x in pp])
            print('path:'   , [x.id for x in p])
            print('-----------------')

            # failed
            if success == 0:
                if requestInfo.state == 0 or requestInfo.state == 1:    # 0, 1
                    self.resetFailedRequestFor01(requestInfo, usedLinks)
                elif requestInfo.state == 2:                            # 2
                    requestInfo.savetime += 1
                    self.resetFailedRequestFor01(requestInfo, usedLinks)
                    if requestInfo.savetime > self.r:   # 超出k儲存時間 重頭送 重設req狀態
                        self.resetFailedRequestFor2(requestInfo, usedLinks)  
                continue
            
            # succeed
            if success > 0:
                if requestInfo.state == 0:      # 0
                    self.totalTime += self.timeSlot - req[2]
                    finishedRequest.append(req)
                    for link in usedLinks:
                        link.clearEntanglement()
                elif requestInfo.state == 1:    # 1
                    self.resetSucceedRequestFor1(requestInfo, usedLinks)
                elif requestInfo.state == 2:    # 2
                    self.resetSucceedRequestFor2(requestInfo, usedLinks)
                    self.totalTime += self.timeSlot - req[2]
                    finishedRequest.append(req)
                continue
            # p5 end
        # p4 end

        for req in finishedRequest:
            self.requestState.pop(req)
        self.srcDstPairs.clear()

        tmpTime = 0
        for req in self.requestState:
            tmpTime += self.timeSlot - req[2]
        print('total time:', self.totalTime + tmpTime)

            
    
if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = MyAlgorithm(topo)
    for i in range(0, 100):
        if i < 50:
            a = sample(topo.nodes, 2)
            s.work([(a[0],a[1])], i)
        else:
            s.work([], i)

    # r=40
    # a=0
    # b=0
    # p1 = 0.03
    # p2 = 0.03
    # for i in range(2,500):
    #     k = i-math.ceil(i/(r+1))
    #     for j in range(1,k+1):
    #         b += math.factorial(i-j-1)//math.factorial(math.ceil(j/r)-1)//math.factorial(i-j-math.ceil(j/r))*pow(p1,math.ceil(j/r))*pow((1-p1),i-j-math.ceil(j/r))*p2*pow((1-p2),j-1)
    #     a += i*b
    #     b = 0

    # print(a)



    # r=40
    # a=0
    # aa = 0
    # b=0
    # p1 = 0.01
    # p2 = 0.02
    # i = 2
    # while(1):

    #     k = i-math.ceil(i/(r+1))
    #     for j in range(1,k+1):
    #         b += math.factorial(i-j-1)//math.factorial(math.ceil(j/r)-1)//math.factorial(i-j-math.ceil(j/r))*pow(p1,math.ceil(j/r))*pow((1-p1),i-j-math.ceil(j/r))*p2*pow((1-p2),j-1)  
    #     aa += a
    #     a += i*b

    #     if aa !=0 and a/aa <= 0.005 :
    #       break

    #     b = 0
    #     i += 1

    # print(a)