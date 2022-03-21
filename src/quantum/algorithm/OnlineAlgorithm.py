
from re import A
import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import PickedPath
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link


class OnlineAlgorithm(AlgorithmBase):

    def __init__(self, topo, allowRecoveryPaths = True):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.name = "Online"
        self.majorPaths = []   
        self.recoveryPaths = {} # {PickedPath: [PickedPath, ...], ...}
        self.allowRecoveryPaths = allowRecoveryPaths

    # P2
    def p2(self):
        self.majorPaths.clear()
        self.recoveryPaths.clear()

        while True: 
            candidates = self.calCandidates(self.srcDstPairs) # candidates -> [PickedPath, ...]   
            sorted(candidates, key=lambda x: x.weight)
            if len(candidates) == 0:
                break
            pick = candidates[-1]   # pick -> PickedPath    
            if pick.weight > 0.0: 
                self.pickAndAssignPath(pick)
            else:
                break

            if self.allowRecoveryPaths:
                self.P2Extra()
            
         
    # 對每個SD-pair找出候選路徑，目前不確定只會找一條還是可以多條
    # 目前缺期望值計算 e method 的實作 實作在Topo.py
    def calCandidates(self, pairs: list): # pairs -> [(Node, Node), ...]
        candidates = []
        for pair in pairs:
            candidate = []
            src, dst = pair
            maxM = max(src.remainingQubits, dst.remainingQubits)
            if maxM == 0:   # not enough qubit
                continue

            for w in range(maxM, 0, -1): # w = maxM, maxM-1, maxM-2, ..., 1
                failNodes = []

                # collect failnodes (they dont have enough Qubits for SDpair in width w)
                for node in self.topo.nodes:
                    if node.remainingQubits < 2 * w and node != src and node != dst:
                        failNodes.append(node)

                edges = {} # edges -> {(Node, Node): [Link, ...], ...}

                # collect edges with links 
                for link in self.topo.links:
                    if not link.assigned and link.n1 not in failNodes and link.n2 not in failNodes:
                        if edges.__contains__((link.n1, link.n2)):
                            edges[(link.n1, link.n2)].append(link)
                        else:
                            edges[(link.n1, link.n2)] = [link]

                neighborsOf = {} # neighborsOf -> {Node: [Node, ...], ...}

                # filter available links satisfy width w
                for edge in edges:
                    links = edges[edge]
                    if len(links) >= w:
                        if neighborsOf.__contains__(edge[0]):
                            neighborsOf[edge[0]].append(edge[1])
                        else:
                            neighborsOf[edge[0]] = [edge[1]]
                        if neighborsOf.__contains__(edge[1]):
                            neighborsOf[edge[1]].append(edge[0])
                        else:
                            neighborsOf[edge[1]] = [edge[0]]
                            
                if (len(neighborsOf[src]) == 0 or len(neighborsOf[dst]) == 0):
                    continue

                prevFromSrc = {}   # prevFromSrc -> {cur: prev}

                def getPathFromSrc(n): 
                    path = []
                    cur = n
                    while (cur != topo.sentinal): 
                        path.insert(0, cur)
                        cur = prevFromSrc[cur]
                    return path
                
                E = {node.id : [-sys.float_info.max, [0.0 for i in range(0,w+1)]] for node in self.topo.nodes}  # E -> {Node id: [Int, [double, ...]], ...}
                q = []  # q -> [(E, Node, Node), ...]

                E[src.id] = [sys.float_info.max, [0.0 for i in range(0,w+1)]]
                q.append((E[src.id][0], src, self.topo.sentinal))
                sorted(q, key=lambda q: q[0])

                # Dijkstra by EXT
                while len(q) != 0:
                    contain = q.pop(-1) # pop the node with the highest E
                    u, prev = contain[1], contain[2]
                    if u in prevFromSrc.keys():
                        continue
                    prevFromSrc[u] = prev

                    # If find the dst add path to candidates
                    if u == dst:        
                        candidate.append(PickedPath(E[dst.id][0], w, getPathFromSrc(dst)))
                        break
                    
                    # Update neighbors of w by E
                    for neighbor in neighborsOf[u]:
                        tmp = E[u.id][1]
                        e = self.topo.e(getPathFromSrc(u) + neighbor, w, tmp)
                        newE = [e, tmp]
                        oldE = E[neighbor.id]

                        if oldE[0] < newE[0]:
                            E[neighbor.id] = newE
                            q.append((E[neighbor.id][0], neighbor, u))
                            sorted(q, key=lambda q: q[0])
                # Dijkstra end

                # 假如此SD-pair在width w有找到path則換找下一個SD-pair 目前不確定是否為此機制
                if len(candidate) > 0:
                    candidates += candidate
                    break
            # for w end      
        # for pairs end
        return candidates

    def pickAndAssignPath(self, pick: PickedPath, majorPath: PickedPath = None):
        if majorPath != None:
            self.recoveryPaths[majorPath].append(pick)
        else:
            self.majorPaths.append(pick)
            self.recoveryPaths[pick] = []
            
        width = pick.width

        for i in range(0, len(pick.path) - 1):
            links = []
            n1, n2 = pick.path[i], pick.path[i+1]

            for link in n1.links:
                if link.contains(n2) and not link.assigned:
                    links.append(link)
            sorted(links, key=lambda q: q.id)

            for i in range(0, width):
                links[i].assignQubits()
                links[i].tryEntanglement() # just display

    def P2Extra(self):
        for majorPath in self.majorPaths:
            p = majorPath.path
            for l in range(1, self.topo.k + 1):
                for i in range(0, len(p) - l):
                    (src, dst) = (p[i], p[i+l])

                    candidates = self.calCandidates([(src, dst)]) # candidates -> [PickedPath, ...]   
                    sorted(candidates, key=lambda x: x.weight)
                    if len(candidates) == 0:
                        continue
                    pick = candidates[-1]   # pick -> PickedPath    
                    if pick.weight > 0.0: 
                        self.pickAndAssignPath(pick, majorPath)

    def p4(self):
        pass

if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    # s = GreedyHopRouting(topo)
    # s.work([(topo.nodes[3],topo.nodes[99])])
    # a = [sys.float_info.max, [0.0 for i in range(0,10)]]
    # print(a)
    # a = PickedPath(0, 1, [])
    # b = []
    # b.append(a)
    # sorted(b, key=lambda x: x.weight)
    # print(b)