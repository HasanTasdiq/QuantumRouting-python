import sys
import random
import heapq
from queue import PriorityQueue
import networkx as nx
from .Node import Node
from .Link import Link

class Topo:

    def __init__(self, G, q, k, a, degree):
        _nodes, _edges, _positions = G.nodes(), list(G.edges()), nx.get_node_attributes(G, 'pos')
        self.nodes = []
        self.links = []
        self.edges = [] # (Node, Node)
        self.q = q
        self.alpha = a
        self.k = k
        self.sentinal = Node(-1, (-1.0, -1.0), -1, self)

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
                    curLen = sys.maxsize
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
        for node in self.nodes:
            for neighbor in _neighbors[node.id]:
                node.neighbors.append(self.nodes[neighbor])

        # for _node in _nodes:
        #     print(_node, _neighbors[_node])

        # print(len(_edges)*2/len(_nodes))

        # Construct Link
        linkId = 0
        for _edge in _edges:
            self.edges.append((self.nodes[_edge[0]], self.nodes[_edge[1]]))
            rand = int(random.random()*5+3)
            for i in range(0, rand):
                link = Link(self, self.nodes[_edge[0]], self.nodes[_edge[1]], False, False, linkId, self.distance(_positions[_edge[0]], _positions[_edge[1]])) 
                self.links.append(link)
                self.nodes[_edge[0]].links.append(link)
                self.nodes[_edge[1]].links.append(link)
                linkId += 1

        # print p and width for test
        p = self.shortestPath(self.nodes[3], self.nodes[99], 'Hop')[1]
        for n in p:
            print(n.id)
        # print('width:', self.widthPhase2(p))
        

    def distance(self, pos1, pos2): # para1 type: tuple, para2 type: tuple
        d = 0
        for a, b in zip(pos1, pos2):
            d += (a-b) ** 2
        return d ** 0.5

    def generate(n, q, k, a, degree):
        dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        G = nx.waxman_graph(n, 0.5, 0.1, L=50/(n**0.5) , domain=(0, 0, 100, 100), metric=dist)

        return Topo(G, q, k, a, degree)

    def widthPhase2(self, path):
        curMinWidth = min(path[0].remainingQubits, path[-1].remainingQubits)

        # Check min qubits in path
        for i in range(1, len(path) - 1):
            if path[i].remainingQubits / 2 < curMinWidth:
                curMinWidth = path[i].remainingQubits // 2

        # Check min links in path
        for i in range(0, len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            t = 0
            for link in n1.links:
                if link.contains(n2) and not link.assigned:
                    t += 1

            if t < curMinWidth:
                curMinWidth = t

        return curMinWidth
        

    def shortestPath(self, src, dst, greedyType):
        # Construct state metric (weight) table for edges
        fStateMetric = {}   # {edge : fstate}
        fStateMetric.clear()
        if greedyType == 'Hop':
            fStateMetric = {edge : 1 for edge in self.edges}
        else: 
            fStateMetric = {edge : self.distance(edge[0].loc, edge[1].loc) for edge in self.edges}

        # Construct neightor & weight table for nodes
        neighborsOf = {}    # {node : {neighbor node : weight}}
        for edge in self.edges:
            n1, n2 = edge
            if neighborsOf.__contains__(n1):
                neighborsOf[n1].update({n2 : fStateMetric[edge]})
            else:
                neighborsOf[n1] = {n2 : fStateMetric[edge]}

            if neighborsOf.__contains__(n2):
                neighborsOf[n2].update({n1 : fStateMetric[edge]})
            else:
                neighborsOf[n2] = {n1 : fStateMetric[edge]}

        D = {node.id : sys.float_info.max for node in self.nodes} 
        q = [] # [(weight, curr, prev)]

        D[src.id] = 0.0
        prevFromSrc = {}   # {cur : prev}

        q.append((D[src.id], src, self.sentinal))
        sorted(q, key=lambda q: q[0])

        # Dijkstra 
        while len(q) != 0:
            contain = q.pop(0)
            w, prev = contain[1], contain[2]
            if w in prevFromSrc.keys():
                continue
            prevFromSrc[w] = prev

            # If find the dst return D & path 
            if w == dst:
                path = []
                cur = dst
                while cur != self.sentinal:
                    path.insert(0, cur)
                    cur = prevFromSrc[cur]          
                return (D[dst.id], path)
            
            # Update neighbors of w  
            for neighbor in neighborsOf[w]:
                weight = neighborsOf[w][neighbor]
                newDist = D[w.id] + weight
                oldDist = D[neighbor.id]

                if oldDist > newDist:
                    D[neighbor.id] = newDist
                    q.append((D[neighbor.id], neighbor, w))
                    sorted(q, key=lambda q: q[0])

        return (sys.float_info.max, [])
        

    def hopsAway(self, src, dst, greedyType):
        # print('enter hopsAway')
        path = self.shortestPath(src, dst, greedyType)
        return len(path[1]) - 1
        
