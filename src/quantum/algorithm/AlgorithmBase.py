import sys
sys.path.append("..")
from topo.Topo import Topo  
import networkx as nx

class AlgorithmBase:

    def __init__(self, topo):
        self.topo = topo
    
    def __prepare(self):
        pass
    
    def __p2(self):
        pass

    def __p4(self):
        pass

    def __tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    def work(self, pairs: list) -> list:  # pair ->　list[tuple(Node, Node)]
        # print("work\n")

        self.__p2()

        self.__tryEntanglement()

        self.__p4()



if __name__ == '__main__':

    dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
    G = nx.waxman_graph(20, 0.5, 0.1, domain=(0, 0, 100, 100), metric=dist)

    edges = G.edges()
    nodes = G.nodes()
    position = nx.get_node_attributes(G, 'pos')
    #print(len(list(edges)))
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    #print(len(edges)*2/600)
    # for i in edges:
    #     print(i)
    # for i in range(0,20):
    #     print(list(nx.neighbors(G,i)))
    
    