from dataclasses import dataclass
from time import process_time, sleep
import sys
sys.path.append("..")
from topo.Topo import Topo  
import time
from topo.helper import needlink_timeslot
from topo.Link import Link

class AlgorithmResult:
    def __init__(self):
        self.algorithmRuntime = 0
        self.waitingTime = 0
        self.idleTime = 0
        self.usedQubits = 0
        self.temporaryRatio = 0
        self.numOfTimeslot = 0
        self.totalRuntime = 0
        self.Ylabels = ["algorithmRuntime", "waitingTime", "idleTime", "usedQubits", "temporaryRatio", 'entanglementPerRound']
        self.remainRequestPerRound = []
        self.usedPaths=[]
        self.entanglementPerRound = []
        self.eps = 0

    def toDict(self):
        dic = {}
        dic[self.Ylabels[0]] = self.algorithmRuntime
        dic[self.Ylabels[1]] = self.waitingTime
        dic[self.Ylabels[2]] = self.idleTime
        dic[self.Ylabels[3]] = self.usedQubits
        dic[self.Ylabels[4]] = self.temporaryRatio
        dic[self.Ylabels[5]] = self.eps

        return dic
    
    def Avg(results: list):
        AvgResult = AlgorithmResult()

        ttime = 201
        AvgResult.remainRequestPerRound = [0 for _ in range(ttime)]
        AvgResult.entanglementPerRound = [0 for _ in range(ttime)]
        for result in results:
            AvgResult.algorithmRuntime += result.algorithmRuntime
            AvgResult.waitingTime += result.waitingTime
            AvgResult.idleTime += result.idleTime
            AvgResult.usedQubits += result.usedQubits
            AvgResult.temporaryRatio += result.temporaryRatio

            Len = len(result.remainRequestPerRound)
            if ttime != Len:
                print("the length of RRPR error:", Len, file = sys.stderr)
            
            for i in range(ttime):
                AvgResult.remainRequestPerRound[i] += result.remainRequestPerRound[i]
                # AvgResult.entanglementPerRound[i] += result.entanglementPerRound[i]
                AvgResult.eps += result.entanglementPerRound[i]


        AvgResult.algorithmRuntime /= len(results)
        AvgResult.waitingTime /= len(results)
        AvgResult.idleTime /= len(results)
        AvgResult.usedQubits /= len(results)
        AvgResult.temporaryRatio /= len(results)
        AvgResult.eps /= len(results)
        AvgResult.eps /= ttime

        for i in range(ttime):
            AvgResult.remainRequestPerRound[i] /= len(results)
            AvgResult.entanglementPerRound[i] /= len(results)
            
        return AvgResult

class AlgorithmBase:

    def __init__(self, topo , preEnt = False , param = None):
        self.name = "Greedy"
        self.topo = topo
        self.srcDstPairs = []
        self.timeSlot = 0
        self.result = AlgorithmResult()
        self.preEnt = preEnt
        self.param = param

    def prepare(self):
        pass
    
    def p2(self):
        pass

    def p4(self):
        pass

    def tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement(self.timeSlot, self.param)
        for segment in self.topo.segments:
            entangled = segment.tryEntanglement(self.timeSlot, self.param)
    # def tryEntanglement(self):
    #     for segment in self.topo.segments:
    #         entangled = segment.tryEntanglement(self.timeSlot, self.param)
    #         # print('entangled ' , entangled , segment.l)
    #         # time.sleep(.001)
    def preEntanglement(self):
        self.topo.preEntanglement()

        # pass
    def updateCacheTable(self):
        for path in self.result.usedPaths:
            # print('len of acc ' , len(path))
            for k in  range(len(path) - 1):
                sd = (path[k] , path[k + 1])
                if sd not in self.topo.cacheTable:
                    self.topo.cacheTable[sd] = 1
                else:
                    self.topo.cacheTable[sd] += 1
        print('self.topo.cacheTable ' , len(self.topo.cacheTable) , 'preEnt ' , self.preEnt)
        count = 0
        for link in self.topo.cacheTable:
            if self.topo.cacheTable[sd] > 1:
                count +=1
        # print('link to generate ent ' , self.topo.cacheTable)
    def tryPreSwapp(self):
        temp_edges = set()
        for link in self.topo.links:
            n1 = link.n1
            n2 = link.n2
            if (n1,n2) not in temp_edges and (n2,n1) not in temp_edges:
                temp_edges.add((n1,n2))

        print('--------------')
        for (node , node1 , node2) in self.topo.needLinksDict:
            if len(self.topo.needLinksDict[(node , node1 , node2)]) <= needlink_timeslot * self.topo.preSwapFraction:
                continue
            if (node1,node2) in temp_edges or (node2,node1) in temp_edges:

                print('========== found existence =========' , node1.id , node2.id, len(self.topo.needLinksDict[(node , node1 , node2)]))
                continue
            
            link1 = None
            link2 = None
            for link in node.links:
                if link.contains(node1) and link.isEntangled(self.timeSlot) and link.notSwapped() and not link.isVirtualLink and link1 is None:
                    link1 = link
                if link.contains(node2) and link.isEntangled(self.timeSlot) and link.notSwapped() and not link.isVirtualLink and link2 is None:
                    link2 = link

            if link1 is not None and link2 is not None:
                # if link1.isVirtualLink or link2.isVirtualLink:
                #     print('!!!!!!!!!!!!!!!!!!!!!!! virtual !!!!!!!!!!!!!!!!')

                link = Link(self.topo, node1, node2, False, False, self.topo.lastLinkId, 0 , isVirtualLink=True)
                # print('if link.assignable() ' , link.assignable())
                if link.assignable():
                    swapped = node.attemptPreSwapping(link1, link2)
                    if swapped:
                        # print('if swapped ' , swapped)
                        
                        link.assignQubits()
                        link.entangled = True
                        link.entangledTimeSlot = min(link1.entangledTimeSlot , link2.entangledTimeSlot)
                        link.subLinks.append(link1)
                        link.subLinks.append(link2)

                        self.topo.addLink(link)

                        self.topo.removeLink(link1)
                        self.topo.removeLink(link2)

                        print('[' , self.name, '] :', self.timeSlot ,  ', == len virtual links ==  :', sum(link.isVirtualLink for link in self.topo.links) )

                        
    
    def resetNodeSwaps(self):
        for node in self.topo.nodes:
            for preInternelLinks in node.prevInternalLinks:
                l1 = preInternelLinks[0]
                l2 = preInternelLinks[1]
                if not (l1.isEntangled(self.timeSlot) and l2.isEntangled(self.timeSlot)):
                    node.prevInternalLinks.remove(preInternelLinks)
    
    def resetNeedLinksDict(self):
        temp = []
        for key in self.topo.needLinksDict:
            slots = self.topo.needLinksDict[key]

            while self.timeSlot - slots[0] > needlink_timeslot:
                slots.pop(0)

                if len(slots) == 0:
                    temp.append(key)
                    break
        for key in temp:
            del self.topo.needLinksDict[key]
        
        self.topo.needLinksDict = dict(sorted(self.topo.needLinksDict.items(), key=lambda item: -len(item[1])))
        

    def work(self, pairs: list, time): 

        self.timeSlot = time # 紀錄目前回合
        self.srcDstPairs.extend(pairs) # 任務追加進去

        if self.timeSlot == 0:
            self.prepare()

        # start
        start = process_time()

        # if self.preEnt:
        #     self.preEntanglement()

        self.p2()
        
        self.tryEntanglement()

        res = self.p4()

        # if self.preEnt:
        #     self.updateCacheTable()

        # end   
        end = process_time()

        self.srcDstPairs.clear()
        self.resetNodeSwaps()
        self.resetNeedLinksDict()

        res.totalRuntime += (end - start)
        res.algorithmRuntime = res.totalRuntime / res.numOfTimeslot

        return res

@dataclass
class PickedPath:
    weight: float
    width: int
    path: list
    time: int

    def __hash__(self):
        return hash((self.weight, self.width, self.path[0], self.path[-1]))

if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    # neighborsOf = {}
    # neighborsOf[1] = {1:2}
    # neighborsOf[1].update({3:3})
    # neighborsOf[2] = {2:1}

    # print(neighborsOf[2][2])
   