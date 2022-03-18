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
        self.pathsSortedDynamically = []

    def _p2(self):
        self.pathsSortedDynamically.clear()

        while True:
            found = False   # record this round whether find new path

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
                    selectedNeighbors = []    # type Node
                    selectedNeighbors.clear()
                    for neighbor in last.neighbors:
                        selectedNeighbors = []
                        if neighbor.remainingQubits > 2 or neighbor == dst and neighbor.remainingQubits > 1:
                            for link in neighbor.links:
                                if link.contains(last) and (not link.assigned):
                                    selectedNeighbors.append(neighbor)

                    # Choose the neighbor with smallest number of hop from it to dst
                    next = self.topo.sentinal
                    hopsCurMinNum = sys.maxsize
                    for selectedNeighbor in selectedNeighbors:
                        hopsNum = Topo.hopsAway(selectedNeighbor, dst, 'Hop') - 1
                        if hopsCurMinNum > hopsNum:
                            hopsCurMinNum = hopsNum
                            next = selectedNeighbor

                    
                    # If have cycle, break
                    if next == topo.sentinal or next in p:
                        break 
                    p.append(next)
                # while end

                if p[-1] != dst:
                    continue
                
                # Caculate width for p
                width = topo.widthPhase2(p)
                
                if width == 0:
                    continue

                found = True
                self.pathsSortedDynamically.append((0.0, width, p))
                sorted(self.pathsSortedDynamically, key=lambda x: x[1])

                # Assign Qubits for links in path 
                for i in range(0, width):
                    for s in range(0, len(p) - 1):
                        n1 = p[s]
                        n2 = p[s+1]
                        for link in n1.links:
                            if link.cintains(n2) and (not link.assigned):
                                link.assignQubits()
                                break    
            # SDpairs end

            if not found:
                break
        # while end
                    
if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
  
