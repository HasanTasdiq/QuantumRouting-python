import sys
sys.path.append("..")
from topo.Topo import Topo  
import networkx as nx

class AlgorithmBase:

    def __init__(self, topo):
        self.name = "Greedy_H"
        self.topo = topo
        self.srcDstPairs = []

    def __prepare(self):
        pass
    
    def __p2(self):
        pass

    def __p4(self):
        pass

    def __tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    def work(self, pairs: list): 
        self.srcDstPairs.clear()
        self.srcDstPairs = pairs

        self.__p2()

        self.__tryEntanglement()

        self.__p4()



if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    