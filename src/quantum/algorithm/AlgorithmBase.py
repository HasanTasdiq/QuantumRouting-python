from dataclasses import dataclass
from time import process_time, sleep
import sys
sys.path.append("..")
from topo.Topo import Topo  

class AlgorithmResult:
    def __init__(self):
        self.algorithmRuntime = 0
        self.waitingTime = 0
        self.unfinishedRequest = 0
        self.idleTime = 0
        self.usedQubits = 0
        self.temporaryRatio = 0
        self.Ylabels = ["algorithmRuntime", "waitingTime", "unfinishedRequest", "idleTime", "usedQubits", "temporaryRatio"]

    def toDict(self):
        dic = {}
        dic[self.Ylabels[0]] = self.algorithmRuntime
        dic[self.Ylabels[1]] = self.waitingTime
        dic[self.Ylabels[2]] = self.unfinishedRequest
        dic[self.Ylabels[3]] = self.idleTime
        dic[self.Ylabels[4]] = self.usedQubits
        dic[self.Ylabels[5]] = self.temporaryRatio
        return dic
    
    def Avg(results: list):
        AvgResult = AlgorithmResult()

        for result in results:
            AvgResult.algorithmRuntime += result.algorithmRuntime
            AvgResult.waitingTime += result.waitingTime
            AvgResult.unfinishedRequest += result.unfinishedRequest
            AvgResult.idleTime += result.idleTime
            AvgResult.usedQubits += result.usedQubits
            AvgResult.temporaryRatio += result.temporaryRatio

        AvgResult.algorithmRuntime /= len(results)
        AvgResult.waitingTime /= len(results)
        AvgResult.unfinishedRequest /= len(results)
        AvgResult.idleTime /= len(results)
        AvgResult.usedQubits /= len(results)
        AvgResult.temporaryRatio /= len(results)

        return AvgResult

class AlgorithmBase:

    def __init__(self, topo):
        self.name = "Greedy"
        self.topo = topo
        self.srcDstPairs = []
        self.timeSlot = 0
        self.result = AlgorithmResult()

    def prepare(self):
        pass
    
    def p2(self):
        pass

    def p4(self):
        pass

    def tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    def work(self, pairs: list, time): 

        self.timeSlot = time # 紀錄目前回合
        self.srcDstPairs.extend(pairs) # 任務追加進去

        if self.timeSlot == 0:
            self.prepare()

        # start
        start = process_time()

        self.p2()
        
        self.tryEntanglement()

        res = self.p4()

        # end   
        end = process_time()

        self.srcDstPairs.clear()

        res.algorithmRuntime += end - start

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
   