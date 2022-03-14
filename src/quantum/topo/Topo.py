import random
import networkx as nx
from .Node import Node
from .Link import Link

class Topo:

    def __init__(self, G, q, k, a, degree):
        _nodes, _edges, _positions = G.nodes(), list(G.edges()), nx.get_node_attributes(G, 'pos')
        self.nodes = []
        self.links = []
        self.q = q
        self.alpha = a
        self.k = k
        
        # Construct neighbor table 
        _neighbors = {}
        for node in _nodes:
            _neighbors[node] = list(nx.neighbors(G,node))
          
        # Construct Node 
        for node in _nodes:
            self.nodes.append(Node(node, _positions[node], random.random()*5+10 , self))   
            usedNode = []
            usedNode.append(node) 
            
            # Make the number of neighbors approach degree  
            if len(_neighbors[node]) < degree - 1:  
                for i in range(0, degree - 1 - len(_neighbors[node])):
                    curNode = -1
                    curLen = 1000000
                    for node2 in _nodes:
                        # print(_positions[node], _positions[node2])
                        if node2 not in usedNode and node2 not in _neighbors[node] and self.distance(_positions[node], _positions[node2]) < curLen:
                            # print(_positions[node], _positions[node2], self.distance(_positions[node], _positions[node2]))
                            curNode = node2
                            curLen = self.distance(_positions[node], _positions[node2])

                    if curNode >= 0:
                        _neighbors[node].append(curNode)
                        _neighbors[curNode].append(node)
                        _edges.append((node, curNode))
                        usedNode.append(curNode)
                        #print(usedNode)

        # for node in _nodes:
        #     print(len(_neighbors[node]))
        #     print(_neighbors[node])

        # Construct Link
        linkId = 0
        print(len(_edges)*2/len(_nodes))
        for edge in _edges:
            # print(edge)
            rand = int(random.random()*5+3)
            for i in range(0,rand): 
                self.links.append(Link(self, self.nodes[edge[0]], self.nodes[edge[1]], False, False, linkId, self.distance(_positions[edge[0]], _positions[edge[1]])))
                linkId += 1
        

    def distance(self, pos1, pos2): # para1 type: tuple, para2 type: tuple
        d = 0
        for a, b in zip(pos1, pos2):
            d += (a-b) ** 2
        return d ** 0.5

    def generate(n, q, k, a, degree):
        dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        G = nx.waxman_graph(n, 0.5, 0.1, L=50/(n**0.5) ,domain=(0, 0, 100, 100), metric=dist)

        return Topo(G, q, k, a, degree)
        
