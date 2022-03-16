import random
import queue
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
        for _node in _nodes:
            _neighbors[_node] = list(nx.neighbors(G,_node))
          
        # Construct Node 
        for _node in _nodes:
            self.nodes.append(Node(_node, _positions[_node], random.random()*5+10 , self))   
            usedNode = []
            usedNode.append(_node) 
            
            # Make the number of neighbors approach degree  
            if len(_neighbors[_node]) < degree - 1:  
                for i in range(0, degree - 1 - len(_neighbors[_node])):
                    curNode = -1
                    curLen = 1000000
                    for _node2 in _nodes:
                        # print(_positions[_node], _positions[_node2])
                        if _node2 not in usedNode and _node2 not in _neighbors[_node] and self.distance(_positions[_node], _positions[_node2]) < curLen:
                            # print(_positions[_node], _positions[_node2], self.distance(_positions[_node], _positions[_node2]))
                            curNode = _node2
                            curLen = self.distance(_positions[_node], _positions[_node2])

                    if curNode >= 0:
                        _neighbors[_node].append(curNode)
                        _neighbors[curNode].append(_node)
                        _edges.append((_node, curNode))
                        usedNode.append(curNode)
                        #print(usedNode)

        # Construct node's neighbor list in Node struct
        for node, _node in self.nodes, _nodes:
            node.neighbors.append(self.nodes[_node])


        # for node in _nodes:
        #     print(len(_neighbors[node]))
        #     print(_neighbors[node])

        print(len(_edges)*2/len(_nodes))

        # Construct Link
        linkId = 0
        for _edge in _edges:
            # print(edge)
            rand = int(random.random()*5+3)
            for i in range(0, rand):
                link = Link(self, self.nodes[_edge[0]], self.nodes[_edge[1]], False, False, linkId, self.distance(_positions[_edge[0]], _positions[_edge[1]])) 
                self.links.append(link)
                self.nodes[_edge[0]].links.append(link)
                self.nodes[_edge[1]].links.append(link)
                linkId += 1
        

    def distance(self, pos1, pos2): # para1 type: tuple, para2 type: tuple
        d = 0
        for a, b in zip(pos1, pos2):
            d += (a-b) ** 2
        return d ** 0.5

    def generate(n, q, k, a, degree):
        dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        G = nx.waxman_graph(n, 0.5, 0.1, L=50/(n**0.5) , domain=(0, 0, 100, 100), metric=dist)

        return Topo(G, q, k, a, degree)
        
