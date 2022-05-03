import threading
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


def run(algo, requests, algoIndex, numOfRequestPerRound, ttime):            
    for i in range(ttime):
        result = algo.work(requests[i], i)
    if algoIndex == 1:
        for req in algo.requestState:
            if algo.requestState[req].state == 2:
                algo.requestState[req].intermediate.clearIntermediate()
    t[numOfRequestPerRound][algo] = result

if __name__ == '__main__':
    global t
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    f = open('../../plot/data/data.txt', 'w')

    algorithms = []
    algorithms.append(GreedyHopRouting(topo))
    algorithms.append(MyAlgorithm(topo))
    algorithms.append(GreedyGeographicRouting(topo))
    algorithms.append(OnlineAlgorithm(topo))
    algorithms.append(REPS(topo))

    numOfRequestPerRoundMax = 2
    t = [{algo : 0 for algo in algorithms} for _ in range(numOfRequestPerRoundMax + 1)]
    ttime = 200
    rtime = 10

    threads = [[] for _ in range(numOfRequestPerRoundMax + 1)]

    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        samplesPerTime = 2 * numOfRequestPerRound
        requests = {i : [] for i in range(ttime)}
        for i in range(ttime):
            if i < rtime:
                a = sample(topo.nodes, samplesPerTime)
                for n in range(0,samplesPerTime,2):
                    requests[i].append((a[n], a[n+1]))
        
        for algoIndex in range(len(algorithms)):
            algo = algorithms[algoIndex]
            Job = threading.Thread(target = run, args = (algo, requests, algoIndex, numOfRequestPerRound, ttime))
            threads[numOfRequestPerRound].append(Job)
            Job.start()

    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        for algoIndex in range(len(algorithms)):
            threads[numOfRequestPerRound][algoIndex].join()

    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        for algoIndex in range(len(algorithms)):
            if algoIndex > 0:
                f.write(' ')
            f.write(str(numOfRequestPerRound))
        f.write('\n')
    # 5XX
    f.close()
    
    