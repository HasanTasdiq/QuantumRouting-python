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

        ttime = 100
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
    def updateNeedLinksDict(self , path):
        upper = len(path)
        if self.name == 'SEER_6':
            upper = 3
        for i in range(2 , upper):
        # for i in range(2 ,3 ):
            segments = self.getSegments(path , i)
            for (node1 , node2) in segments:
                if (node1 , node2) in self.topo.needLinksDict:
                    self.topo.needLinksDict[(node1 , node2)].append(self.timeSlot)
                elif (node2 , node1) in self.topo.needLinksDict:
                    self.topo.needLinksDict[(node2 , node1)].append(self.timeSlot)
                else:
                    self.topo.needLinksDict[(node1 , node2)] = ([self.timeSlot])


    def getSegments(self , a , n):
        res = []
        for i in range(len(a) - n):
            res.append((a[i] , a[i + n]))
        return res
    
    # def tryPreSwapp(self):
    #     temp_edges = set()
    #     for link in self.topo.links:
    #         n1 = link.n1
    #         n2 = link.n2
    #         if (n1,n2) not in temp_edges and (n2,n1) not in temp_edges:
    #             temp_edges.add((n1,n2))

    #     print('--------------')
    #     for (node , node1 , node2) in self.topo.needLinksDict:
    #         if len(self.topo.needLinksDict[(node , node1 , node2)]) <= needlink_timeslot * self.topo.preSwapFraction:
    #             continue

    #         # if (node1,node2) in temp_edges or (node2,node1) in temp_edges:
    #         #     print('========== found existence =========' , node1.id , node2.id, len(self.topo.needLinksDict[(node , node1 , node2)]))
    #             # continue

    #         nodeVirtualLink = len([link for link in node1.links if (link.isVirtualLink and link.contains(node2))])
    #         path = [node1 , node , node2]
    #         if self.topo.widthPhase2(path) < 2:
    #             continue

    #         # if nodeVirtualLink >= 1:
    #         #     continue
    #         # continue
            
    #         link1 = None
    #         link2 = None
    #         for link in node.links:
    #             if link.contains(node1) and link.isEntangled(self.timeSlot) and link.notSwapped() and not link.isVirtualLink and link1 is None:
    #                 link1 = link
    #             if link.contains(node2) and link.isEntangled(self.timeSlot) and link.notSwapped() and not link.isVirtualLink and link2 is None:
    #                 link2 = link

    #         if link1 is not None and link2 is not None:
    #             # if link1.isVirtualLink or link2.isVirtualLink:
    #             #     print('!!!!!!!!!!!!!!!!!!!!!!! virtual !!!!!!!!!!!!!!!!')

    #             link = Link(self.topo, node1, node2, False, False, self.topo.lastLinkId, 0 , isVirtualLink=True)
    #             # print('if link.assignable() ' , link.assignable())
    #             if link.assignable():
    #                 swapped = node.attemptPreSwapping(link1, link2)
    #                 if swapped:
    #                     # print('if swapped ' , swapped)
                        
    #                     # link.assignQubits()
    #                     link.entangled = True
    #                     link.entangledTimeSlot = min(link1.entangledTimeSlot , link2.entangledTimeSlot)
    #                     link.subLinks.append(link1)
    #                     link.subLinks.append(link2)

    #                     self.topo.addLink(link)

    #                     self.topo.removeLink(link1)
    #                     self.topo.removeLink(link2)

    #     print('[' , self.name, '] :', self.timeSlot ,  ', == len virtual links ==  :', sum(link.isVirtualLink for link in self.topo.links) )



    def tryPreSwapp(self):
        temp_edges = set()
        for link in self.topo.links:
            n1 = link.n1
            n2 = link.n2
            if (n1,n2) not in temp_edges and (n2,n1) not in temp_edges:
                temp_edges.add((n1,n2))

        print('--------------')
        # for (source , dest) in self.topo.needLinksDict:
        #     if len(self.topo.needLinksDict[(source , dest)]) >= needlink_timeslot * self.topo.preSwapFraction:

        #         print('***===****src: ' , source.id , 'dest: ' , dest.id , '-', len(self.topo.needLinksDict[(source , dest)]),'==' ,  self.topo.hopsAway(source , dest , 'hop') )

        for (source , dest) in self.topo.needLinksDict:

            if len(self.topo.needLinksDict[(source , dest)]) <= needlink_timeslot * self.topo.preSwapFraction:
                continue
            print('***===****src-dest: ' , (source.id ,dest.id ), '=', len(self.topo.needLinksDict[(source , dest)]),'==' ,  self.topo.hopsAway(source , dest , 'hop') )

            # if (node1,node2) in temp_edges or (node2,node1) in temp_edges:
            #     print('========== found existence =========' , node1.id , node2.id, len(self.topo.needLinksDict[(node , node1 , node2)]))
                # continue
            k = 5
            if self.name == 'SEER_6':
                
                k = 1
            paths = self.topo.k_alternate_paths(source.id , dest.id , k)
            for path in paths:
                # print('** path len ' , len(path))
                

                path2 = [self.topo.nodes[nodeId] for nodeId in path]
                # print([n for n in path])
                if self.topo.widthPhase2(path2) < 2:
                    print('self.topo.widthPhase2(path2) ' , self.topo.widthPhase2(path2))
                    continue

                for i in range(1 , len(path) - 1):
                    node1 = self.topo.nodes[path[0]]
                    node = self.topo.nodes[path[i]]
                    node2 = self.topo.nodes[path[i + 1]]

            
                    link1 = None
                    link2 = None
                    for link in node.links:
                        # if link.contains(node1) or link.contains(node2):
                        #     print('=!=!=======link.isEntangled:' , link.isEntangled(self.timeSlot) , 'link.notSwa:' , link.notSwapped() , 'link1None:', link1 is None , 'link2 is None:',link2 is None)
                        if link.contains(node1) and link.isEntangled(self.timeSlot) and link.notSwapped() and link1 is None:
                            link1 = link
                        if link.contains(node2) and link.isEntangled(self.timeSlot) and link.notSwapped() and link2 is None:
                            link2 = link

                    if link1 is not None and link2 is not None:
                        # if link1.isVirtualLink or link2.isVirtualLink:
                        #     print('!!!!!!!!!!!!!!!!!!!!!!! virtual !!!!!!!!!!!!!!!!')

                        link = Link(self.topo, node1, node2, False, False, self.topo.lastLinkId, 0 , isVirtualLink=True)
                        print('if link.assignable() ', (source.id ,dest.id ) , link.assignable())
                        if link.assignable():
                            swapped = node.attemptPreSwapping(link1, link2)
                            # print('if swapped ' , swapped)

                            if swapped:
                                
                                # link.assignQubits()
                                link.entangled = True
                                link.entangledTimeSlot = min(link1.entangledTimeSlot , link2.entangledTimeSlot)
                                if link1.isVirtualLink:
                                    link.subLinks.extend(link1.subLinks)
                                else:
                                    link.subLinks.append(link1)

                                if link2.isVirtualLink:
                                    link.subLinks.extend(link2.subLinks)
                                else:
                                    link.subLinks.append(link2)

                                self.topo.addLink(link)

                                self.topo.removeLink(link1)
                                self.topo.removeLink(link2)

                                if link.n1 == source and link.n2 == dest:
                                    print('================complete link created====================' , len(path) , (source.id , dest.id))
                                    if len(path) >3:
                                        print([n for n in path])

        print('[' , self.name, '] :', self.timeSlot ,  ', == len virtual links ==  :', sum(link.isVirtualLink for link in self.topo.links) )



        

                        
    def updateNeighbors(self):
        for node in self.topo.nodes:
            node.neighbors = []
            for link in node.links:
                neighbor = link.theOtherEndOf(node)
                if neighbor not in node.neighbors:
                    node.neighbors.append(neighbor)
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
        
        self.topo.needLinksDict = dict(sorted(self.topo.needLinksDict.items(), key=lambda item: -len(item[1]) * self.topo.hopsAway(item[0][0] , item[0][1] , 'hop')))
        

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
   