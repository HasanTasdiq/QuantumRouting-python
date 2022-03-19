import sys
sys.path.append("..")
from topo.Topo import Topo  

class AlgorithmBase:

    def __init__(self, topo):
        self.name = "Greedy"
        self.topo = topo
        self.srcDstPairs = []

    def prepare(self):
        pass
    
    def p2(self):
        pass

    def p4(self):
        pass

    def tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    def work(self, pairs: list): 
        self.srcDstPairs.clear()
        self.srcDstPairs = pairs
       
        self.p2()
        
        self.tryEntanglement()

        self.p4()
        


if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    # neighborsOf = {}
    # neighborsOf[1] = {1:2}
    # neighborsOf[1].update({3:3})
    # neighborsOf[2] = {2:1}

    # print(neighborsOf[2][2])
   