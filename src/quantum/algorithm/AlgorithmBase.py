from dataclasses import dataclass
import sys
sys.path.append("..")
from topo.Topo import Topo  

class AlgorithmBase:

    def __init__(self, topo):
        self.name = "Greedy"
        self.topo = topo
        self.srcDstPairs = []
        self.finishedSrcDstPairs = []
        self.timeSlot = 0

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
        # self.finishedSrcDstPairs.clear()
        self.timeSlot = time # 紀錄目前回合
        self.srcDstPairs.extend(pairs) # 任務追加進去

        if self.timeSlot == 0:
            self.prepare()

        self.p2()
        
        self.tryEntanglement()

        self.p4()

        self.srcDstPairs.clear()

@dataclass
class PickedPath:
    weight: float
    width: int
    path: list

    def __hash__(self):
        return hash((self.weight, self.width, self.path[0], self.path[-1]))

if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    # neighborsOf = {}
    # neighborsOf[1] = {1:2}
    # neighborsOf[1].update({3:3})
    # neighborsOf[2] = {2:1}

    # print(neighborsOf[2][2])
   