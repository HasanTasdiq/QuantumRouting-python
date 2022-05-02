import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from MyAlgorithm import MyAlgorithm
from OnlineAlgorithm import OnlineAlgorithm
from GreedyGeographicRouting import GreedyGeographicRouting
from GreedyHopRouting import GreedyHopRouting
from REPS import REPS
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from random import sample
if __name__ == '__main__':

    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    f = open('../../plot/data/data.txt', 'w')

    algo = []
    algo.append(GreedyHopRouting(topo))
    algo.append(MyAlgorithm(topo))
    algo.append(GreedyGeographicRouting(topo))
    algo.append(OnlineAlgorithm(topo))
    algo.append(REPS(topo))

    numOfAlgo = len(algo)
    samplesPerTime = 2

    while samplesPerTime < 11:
        ttime = 200
        rtime = 10
        requests = {i : [] for i in range(ttime)}
        t = {a : 0 for a in algo}

        f.write(str(samplesPerTime/2)+' ')
        f.flush()
        for i in range(ttime):
            if i < rtime:
                a = sample(topo.nodes, samplesPerTime)
                for n in range(0,samplesPerTime,2):
                    requests[i].append((a[n], a[n+1]))
            

        for algoIndex in range(numOfAlgo):
            a = algo[algoIndex]
            for i in range(ttime):
                t[a] = a.work(requests[i], i)
            if algoIndex == 1:
                for req in a.requestState:
                    if a.requestState[req].state == 2:
                        a.requestState[req].intermediate.clearIntermediate()    
                
            if algoIndex > 0:
                f.write(' ')
            f.write(str(t[a]/(samplesPerTime/2*rtime)))
            f.flush()
        f.write('\n')
        samplesPerTime += 2 

    # 5XX
    f.close()
    
    