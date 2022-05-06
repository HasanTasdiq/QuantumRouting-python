import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from random import sample

class GreedyGeographicRouting(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.requests = []
        self.totalTime = 0
        self.name = "Greedy_G"

    def prepare(self):
        self.totalTime = 0
        self.requests.clear()

    def p2(self):
        self.pathsSortedDynamically.clear()

        for req in self.srcDstPairs:
            (src, dst) = req
            self.requests.append((src, dst, self.timeSlot))

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
                    disCurMin = sys.maxsize
                    for selectedNeighbor in selectedNeighbors:
                        dis = self.topo.distance(selectedNeighbor.loc, dst.loc)      
                        if disCurMin > dis:
                            disCurMin = dis
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
                                link.assignQubits()
                                break    
            # for SDpairs end

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
        print('p2 end')
    
    def p4(self):
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
        
            if succ > 0 or len(p) == 2:
                print('finish time:', self.timeSlot - time)
                find = (p[0], p[-1], time)
                if find in self.requests:
                    self.totalTime += self.timeSlot - time
                    self.requests.remove(find)
            print('----------------------')

        remainTime = 0
        for req in self.requests:
            remainTime += self.timeSlot - req[2]

        self.topo.clearAllEntanglements()                     
        self.result.waitingTime = self.totalTime + remainTime

        print('waiting time:', self.result.waitingTime)
        print('idle time:', self.result.idleTime)
        print('p4 end')

        return self.result
                    
if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = GreedyGeographicRouting(topo)
    for i in range(0, 200):
        requests = []
        if i < 10:
            a = sample(topo.nodes, 6)
            for n in range(0,6,2):
                requests.append((a[n], a[n+1]))
            s.work(requests, i)
        else:
            s.work([], i)
    
  