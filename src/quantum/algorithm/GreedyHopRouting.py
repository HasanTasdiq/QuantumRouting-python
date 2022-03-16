from asyncio.windows_events import NULL
import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
import networkx as nx
import queue

class GreedyHopRouting(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.pathsSortedDynamically = {}

    def _p2(self):
        self.pathsSortedDynamically.clear()
    
        # Find the shortest path and assing qubits for every srcDstPair
        for srcDstPair in self.srcDstPairs:
            (src, dst) = srcDstPair
            p = []
            p.append(src)
            
            # Find a shortest path by greedy min hop  
            while True:
                last = p[-1]
                if last == dst:
                    break
                
                # Select avaliable neighbors of last(local)
                selectNeighbors = []    # type Node
                selectNeighbors.clear()
                for neighbor in last.neighbors:
                    selectNeighbors = []
                    if neighbor.remainingQubits > 2 or neighbor == dst and neighbor.remainingQubits > 1:
                        for link in neighbor.links:
                            if link.contains(last) and (not link.assigned):
                                selectNeighbors.append(neighbor)

                # Choose the neighbor with smallest number of hop to dst
                next = 0
                # ...
                 
                # If have cycle, break
                if next == NULL or next in p:
                    break 
                p.append(next)
            # while end

            if p[-1] != dst:
                continue
            
            # Caculate width
            width = 0
            # ...

            if width == 0:
                continue

            # self.pathsSortedDynamically.









if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
  
