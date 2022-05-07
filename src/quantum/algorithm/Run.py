import threading
import sys
import copy
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from MyAlgorithm import MyAlgorithm
from OnlineAlgorithm import OnlineAlgorithm
from GreedyGeographicRouting import GreedyGeographicRouting
from GreedyHopRouting import GreedyHopRouting
from REPS import REPS
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link
from random import sample


def runThread(algo, requests, algoIndex, ttime):
    global results
    for i in range(ttime):
        result = algo.work(requests[i], i)
    if algoIndex == 0:
        for req in algo.requestState:
            if algo.requestState[req].state == 2:
                algo.requestState[req].intermediate.clearIntermediate()
    results[algoIndex].append(result)

def Run(numOfRequestPerRound = 2, numOfNode = 100, r = 40, q = 0.9, alpha = 0.05, density = 0.5, mapSize = (50, 200), rtime = 20):
    global results
    topo = Topo.generate(numOfNode, q, 5, alpha, 6)

    # make copy
    algorithms = []
    algorithms.append(MyAlgorithm(copy.deepcopy(topo)))
    algorithms.append(GreedyHopRouting(copy.deepcopy(topo)))
    algorithms.append(GreedyGeographicRouting(copy.deepcopy(topo)))
    algorithms.append(OnlineAlgorithm(copy.deepcopy(topo)))
    algorithms.append(REPS(copy.deepcopy(topo)))

    algorithms[0].r = r
    algorithms[0].density = density

    times = 2
    results = [[] for _ in range(len(algorithms))]
    ttime = 10


    for _ in range(times):
        
        threads = []
        ids = {i : [] for i in range(ttime)}
        for i in range(ttime):
            if i < rtime:
                for _ in range(numOfRequestPerRound):
                    a = sample([i for i in range(numOfNode)], 2)
                    ids[i].append((a[0], a[1]))
        
        for algoIndex in range(len(algorithms)):
            algo = copy.deepcopy(algorithms[algoIndex])
            requests = {i : [] for i in range(ttime)}
            for i in range(rtime):
                for (src, dst) in ids[i]:
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst]))
            Job = threading.Thread(target = runThread, args = (algo, requests, algoIndex, ttime))
            threads.append(Job)

        for algoIndex in range(len(algorithms)):
            threads[algoIndex].start()


        for algoIndex in range(len(algorithms)):
            threads[algoIndex].join()

    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(results[algoIndex])

    # results[0] = result of GreedyHopRouting = a AlgorithmResult
    # results[1] = result of MyAlgorithm
    # results[2] = result of GreedyGeographicRouting
    # results[3] = result of OnlineAlgorithm
    # results[4] = result of REPS

    return results
    

if __name__ == '__main__':
    print("start")
    targetFilePath = "../../plot/data/"
    temp = AlgorithmResult()
    Ylabels = temp.Ylabels
    # Ylabels = ["algorithmRuntime", "waitingTime", "unfinishedRequest", "idleTime", "usedQubits", "temporaryRatio"]
    
    numOfNodes = [10, 20, 50, 100]
    Xlabel = "numOfnodes"
    Ydata = []
    for numOfNode in numOfNodes:
        result = Run(numOfNode = numOfNode, rtime=2)
        Ydata.append(result)

    # Ydata[0] = numOfNode = 10 algo1Result algo2Result ... 
    # Ydata[1] = numOfNode = 20 algo1Result algo2Result ... 
    # Ydata[2] = numOfNode = 50 algo1Result algo2Result ... 
    # Ydata[3] = numOfNode = 100 algo1Result algo2Result ... 

    for Ylabel in Ylabels:
        filename = Xlabel + "_" + Ylabel + ".txt"
        F = open(targetFilePath + filename, "w")
        for i in range(len(numOfNodes)):
            Xaxis = str(numOfNodes[i])
            Yaxis = [algoResult.toDict()[Ylabel] for algoResult in Ydata[i]]
            Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
            F.write(Xaxis + Yaxis)
