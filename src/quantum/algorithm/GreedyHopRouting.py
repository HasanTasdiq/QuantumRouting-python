import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from MyAlgorithm import MyAlgorithm
from OnlineAlgorithm import OnlineAlgorithm
from GreedyGeographicRouting import GreedyGeographicRouting
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from random import sample

class GreedyHopRouting(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.requests = []
        self.totalTime = 0
        self.totalUsedQubits = 0
        self.totalNumOfReq = 0
        self.name = "Greedy_H"

    def prepare(self):
        self.totalTime = 0
        self.requests.clear()
        
    def p2(self):
        self.pathsSortedDynamically.clear()

        for req in self.srcDstPairs:
            (src, dst) = req
            self.totalNumOfReq += 1
            self.requests.append((src, dst, self.timeSlot))
        
        if len(self.requests) > 0:
            self.result.numOfTimeslot += 1

        while True:
            found = False   # record this round whether find new path

            # Find the shortest path and assing qubits for every srcDstPair
            for req in self.requests:
                (src, dst, time) = req
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
                        if neighbor.remainingQubits > 2 or (neighbor == dst and neighbor.remainingQubits > 1):
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
                        if hopsCurMinNum > hopsNum and hopsNum != -1:
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

                found = True
                self.pathsSortedDynamically.append((0.0, width, p, time))
                sorted(self.pathsSortedDynamically, key=lambda x: x[1])

                # Assign Qubits for links in path 
                for _ in range(0, width):
                    for s in range(0, len(p) - 1):
                        n1 = p[s]
                        n2 = p[s+1]
                        for link in n1.links:
                            if link.contains(n2) and (not link.assigned):
                                self.totalUsedQubits += 2
                                link.assignQubits()
                                break    
            # SDpairs end

            if not found:
                break
        # while end
        for req in self.requests:
            pick = False
            for path in self.pathsSortedDynamically:
                _, width, p, time = path
                if (p[0], p[-1], time) == req:
                    pick = True
                    break               
            if not pick:
                self.result.idleTime += 1
        print('[Greedy_H] p2 end')
    
    def p4(self):
        totalSuccess = 0
        totalEnt = 0
        for path in self.pathsSortedDynamically:
            _, width, p, time = path
            oldNumOfPairs = len(self.topo.getEstablishedEntanglements(p[0], p[-1]))
            
            for i in range(1, len(p) - 1):
                prev = p[i-1]
                curr = p[i]
                next = p[i+1]
                prevLinks = []
                nextLinks = []
                
                w = width
                for link in curr.links:
                    if link.entangled and (link.n1 == prev and not link.s2 or link.n2 == prev and not link.s1) and w > 0:
                        prevLinks.append(link)
                        w -= 1

                w = width
                for link in curr.links:
                    if link.entangled and (link.n1 == next and not link.s2 or link.n2 == next and not link.s1) and w > 0:
                        nextLinks.append(link)
                        w -= 1

                if prevLinks == None or nextLinks == None:
                    break

                for (l1, l2) in zip(prevLinks, nextLinks):
                    curr.attemptSwapping(l1, l2)

            print('----------------------')
            print('path:', [x.id for x in p])
            succ = len(self.topo.getEstablishedEntanglements(p[0], p[-1])) - oldNumOfPairs
            totalEnt += succ
            if succ > 0 or len(p) == 2:
                print('finish time:', self.timeSlot - time)
                find = (p[0], p[-1], time)
                if find in self.requests:
                    self.totalTime += self.timeSlot - time
                    self.requests.remove(find)
                    totalSuccess +=1

            print('----------------------')

        remainTime = 0
        for req in self.requests:
            # self.result.unfinishedRequest += 1
            remainTime += self.timeSlot - req[2]

        self.topo.clearAllEntanglements()
        self.result.remainRequestPerRound.append(len(self.requests)/self.totalNumOfReq)     
        self.result.waitingTime = (self.totalTime + remainTime) / self.totalNumOfReq + 1
        self.result.usedQubits = self.totalUsedQubits / self.totalNumOfReq
        self.result.successfulRequestPerRound.append(totalSuccess)
        self.result.entanglementPerRound.append(totalEnt)
        self.result.successfulRequest += totalSuccess


        print('[Greedy_H] total successful :', totalSuccess)
        print('[Greedy_H] waiting time:', self.result.waitingTime)
        print('[Greedy_H] idle time:', self.result.idleTime)
        print('[Greedy_H] p4 end')

        return self.result
        
if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.0001, 6)
    # f = open('logfile.txt', 'w')
    
    a1 = GreedyHopRouting(topo)
    # a2 = MyAlgorithm(topo)
    # a3 = GreedyGeographicRouting(topo)
    a4 = OnlineAlgorithm(topo)
    # samplesPerTime = 2

    # while samplesPerTime < 11:
    #     ttime = 200
    #     rtime = 10
    #     requests = {i : [] for i in range(ttime)}
    #     t1 = 0
    #     t2 = 0
    #     t3 = 0
    #     t4 = 0
    #     f.write(str(samplesPerTime/2)+' ')
    #     f.flush()
    #     for i in range(ttime):
    #         if i < rtime:
    #             a = sample(topo.nodes, samplesPerTime)
    #             for n in range(0,samplesPerTime,2):
    #                 requests[i].append((a[n], a[n+1]))
            

    #     for i in range(ttime):
    #         t1 = a1.work(requests[i], i)
    #     f.write(str(t1/(samplesPerTime/2*rtime))+' ')
    #     f.flush()

    #     for i in range(ttime):
    #         t3 = a3.work(requests[i], i)
    #     f.write(str(t3/(samplesPerTime/2*rtime))+' ')
    #     f.flush()

    #     for i in range(ttime):
    #         t4 = a4.work(requests[i], i)
    #     f.write(str(t4/(samplesPerTime/2*rtime))+' ')
    #     f.flush()

    #     for i in range(ttime):
    #         t2 = a2.work(requests[i], i)
    #     for req in a2.requestState:
    #         if a2.requestState[req].state == 2:
    #             a2.requestState[req].intermediate.clearIntermediate()    

    #     f.write(str(t2/(samplesPerTime/2*rtime))+'\n')
    #     f.flush()
    #     samplesPerTime += 2 

    # # 5XX
    # f.close()
    
    samplesPerTime = 4
    ttime = 100
    rtime = 10
    requests = {i : [] for i in range(ttime)}

    for i in range(ttime):
        if i < rtime:
            a = sample(topo.nodes, samplesPerTime)
            for n in range(0,samplesPerTime,2):
                requests[i].append((a[n], a[n+1]))

    for i in range(ttime):
        t3 = a4.work(requests[i], i)
    
    for i in range(ttime):
        t1 = a1.work(requests[i], i)