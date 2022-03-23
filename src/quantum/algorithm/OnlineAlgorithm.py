
from dataclasses import dataclass
from re import A
import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import PickedPath
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link

@dataclass
class RecoveryPath:
    path: list
    width: int
    taken: int 
    available: int 


class OnlineAlgorithm(AlgorithmBase):

    def __init__(self, topo, allowRecoveryPaths = True):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.name = "Online"
        self.majorPaths = []    # [PickedPath, ...]
        self.recoveryPaths = {} # {PickedPath: [PickedPath, ...], ...}
        self.pathToRecoveryPaths = {} # {PickedPath : [RecoveryPath, ...], ...}
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
    # ***目前缺期望值計算 e method 的實作 實作在Topo.py
    def calCandidates(self, pairs: list): # pairs -> [(Node, Node), ...]
        candidates = []
        for pair in pairs:
            candidate = []
            src, dst = pair[0], pair[1]
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
                    if link.n2.id < link.n1.id:
                        link.n1, link.n2 = link.n2, link.n1
                    if not link.assigned and link.n1 not in failNodes and link.n2 not in failNodes:
                        if not edges.__contains__((link.n1, link.n2)):
                            edges[(link.n1, link.n2)] = []
                        edges[(link.n1, link.n2)].append(link)

                neighborsOf = {} # neighborsOf -> {Node: [Node, ...], ...}
                neighborsOf[src] = []
                neighborsOf[dst] = []

                # filter available links satisfy width w
                for edge in edges:
                    links = edges[edge]
                    if len(links) >= w:
                        if not neighborsOf.__contains__(edge[0]):
                            neighborsOf[edge[0]] = []
                        if not neighborsOf.__contains__(edge[1]):
                            neighborsOf[edge[1]] = []

                        neighborsOf[edge[0]].append(edge[1])
                        neighborsOf[edge[1]].append(edge[0])
                        neighborsOf[edge[0]] = list(set(neighborsOf[edge[0]]))
                        neighborsOf[edge[1]] = list(set(neighborsOf[edge[1]]))
                
                # test
                for node in neighborsOf:
                    for neigh in neighborsOf[node]:
                        if(node.id == src.id or node.id == dst.id):
                            print(w, node.id, neigh.id)
                                
                if (len(neighborsOf[src]) == 0 or len(neighborsOf[dst]) == 0):
                    continue

                prevFromSrc = {}   # prevFromSrc -> {cur: prev}

                def getPathFromSrc(n): 
                    path = []
                    cur = n
                    while (cur != self.topo.sentinal): 
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
                    
                    # Update neighbors by EXT
                    for neighbor in neighborsOf[u]:
                        tmp = E[u.id][1]
                        p = getPathFromSrc(u)
                        e = self.topo.e(p.append(neighbor), w, tmp)
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
        for pathWithWidth in self.majorPaths:
            width = pathWithWidth.width
            majorPath = pathWithWidth.path

            recoveryPaths = self.recoveryPaths[pathWithWidth]   # recoveryPaths -> [pickedPath, ...]
            sorted(recoveryPaths, key=lambda x: len(x.path)*10000 + majorPath.index(x.path[0])) # sort recoveryPaths by it recoverypath length and the index of the first node in recoveryPath  

            # Construct pathToRecoveryPaths table
            for recoveryPath in recoveryPaths:
                w = recoveryPath.width
                p = recoveryPath.path
                available = sys.maxsize
                for i in range(0, len(p) - 1):
                    n1 = p[i]
                    n2 = p[i+1]
                    cnt = 0
                    for link in n1.links:
                        if link.contains(n2) and link.entangled:
                            cnt += 1
                    if cnt < available:
                        available = cnt
                
                if not self.pathToRecoveryPaths.__contains__(pathWithWidth):
                    self.pathToRecoveryPaths[pathWithWidth] = []
                
                self.pathToRecoveryPaths[pathWithWidth].append(RecoveryPath(p, w, 0, available))
            # for end 

            rpToWidth = {recoveryPath.path: recoveryPath.width for recoveryPath in recoveryPaths}  # rpToWidth -> {pickedPath: int, ...}

            # for w-width major path, treat it as w different paths, and repair separately
            for w in range(1, width + 1):
                brokenEdges = []    # [(int, int), ...]

                # find all broken edges on the major path
                for i in range(0, len(majorPath) - 1):
                    i1 = i
                    i2 = i+1
                    n1 = majorPath[i1]
                    n2 = majorPath[i2]
                    for link in n1.links:
                        if link.contains(n2) and link.assigned and link.notSwapped() and not link.entangledand:
                            brokenEdges.append((i1, i2))
                            continue
                
                edgeToRps = {brokenEdge: [] for brokenEdge in brokenEdges}
                rpToEdges = {recoveryPath.path: [] for recoveryPath in recoveryPaths}

                # Construct edgeToRps & rpToEdges
                for recoveryPath in recoveryPaths:
                    rp = recoveryPath.path
                    s1, s2 = majorPath.index(rp[0]), majorPath.index(rp[-1])

                    for j in range(s1, s2):
                        if (j, j+1) in brokenEdges:
                            edgeToRps[(j, j+1)].append(rp)
                            rpToEdges[rp].append((j, j+1))

                realPickedRps = set()
                realRepairedEdges = set()

                # try to cover the broken edges
                for brokenEdge in brokenEdges:
                    # if the broken edge is repaired, go to repair the next broken edge
                    if brokenEdge in realRepairedEdges: 
                        continue
                    repaired = False
                    next = 0
                    rps = edgeToRps[brokenEdge] 

                    # filter the avaliable rp in rps for brokenEdge
                    for rp in rps:
                        if not (rpToWidth[rp] > 0 and rp not in realPickedRps):
                            rps.remove(rp)

                    # sort rps by the start id in majorPath
                    sorted(rps, key=lambda x: majorPath.index(x[0]) * 10000 + majorPath.index(x[-1]) )

                    for rp in rps:
                        if majorPath.index(rp[0]) < next:
                            continue 

                        next = majorPath.index(rp[-1])
                        pickedRps = realPickedRps
                        repairedEdges = realRepairedEdges
                        otherCoveredEdges = set(rpToEdges[rp]) - {brokenEdge}
                        covered = False

                        for edge in otherCoveredEdges: #delete covered rps, or abort
                            prevRp = set(edgeToRps[edge]) & pickedRps    # 這個edge 所覆蓋到的rp 假如已經有被選過 表示她被修理過了 表示目前這個rp要修的edge蓋到以前的rp
                            
                            if prevRp == set()  :
                                repairedEdges.add(edge)
                            else: 
                                covered = True
                                break  # the rps overlap. taking time to search recursively. just abort
                        
                        if covered:
                            continue

                        repaired = True      
                        repairedEdges.add(brokenEdge) 
                        pickedRps.add(rp)

                        for rp in realPickedRps - pickedRps:
                            rpToWidth[rp] += 1
                        for rp in pickedRps - realPickedRps:
                            rpToWidth[rp] -= 1
                        
                        realPickedRps = pickedRps
                        realRepairedEdges = repairedEdges
                    # for rp end

                    if not repaired:   # this major path cannot be repaired
                        break
                # for brokenEdge end
            # for w end
        # for pathWithWidth end

if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    s = OnlineAlgorithm(topo)
    s.work([(topo.nodes[3],topo.nodes[99])])
    
    a = {1,2,3}
    b = {1,2,3}

    for aa in a-b:
        print(aa)
    print(a&b)
    
    