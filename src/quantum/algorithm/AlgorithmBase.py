from dataclasses import dataclass
from time import process_time, sleep
import sys
sys.path.append("..")
from topo.Topo import Topo  

class AlgorithmResult:
    def __init__(self):
        self.algorithmRuntime = 0
        self.waitingTime = 0
        self.idleTime = 0
        self.usedQubits = 0
        self.temporaryRatio = 0
        self.numOfTimeslot = 0
        self.totalRuntime = 0
        self.Ylabels = ["algorithmRuntime", "waitingTime", "idleTime", "usedQubits", "temporaryRatio", 'remainRequestPerRound']
        self.remainRequestPerRound = []
        self.usedPaths=[]

    def toDict(self):
        dic = {}
        dic[self.Ylabels[0]] = self.algorithmRuntime
        dic[self.Ylabels[1]] = self.waitingTime
        dic[self.Ylabels[2]] = self.idleTime
        dic[self.Ylabels[3]] = self.usedQubits
        dic[self.Ylabels[4]] = self.temporaryRatio
        dic[self.Ylabels[5]] = self.remainRequestPerRound
        return dic
    
    def Avg(results: list):
        AvgResult = AlgorithmResult()

        ttime = 101
        AvgResult.remainRequestPerRound = [0 for _ in range(ttime)]
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


        AvgResult.algorithmRuntime /= len(results)
        AvgResult.waitingTime /= len(results)
        AvgResult.idleTime /= len(results)
        AvgResult.usedQubits /= len(results)
        AvgResult.temporaryRatio /= len(results)

        for i in range(ttime):
            AvgResult.remainRequestPerRound[i] /= len(results)
            
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
   