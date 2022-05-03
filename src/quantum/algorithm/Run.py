import threading
import sys
import copy
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from MyAlgorithm import MyAlgorithm
from OnlineAlgorithm import OnlineAlgorithm
from GreedyGeographicRouting import GreedyGeographicRouting
from GreedyHopRouting import GreedyHopRouting
# from REPS import REPS
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link
from random import sample


def run(algo, requests, algoIndex, numOfRequestPerRound, ttime):
    global t
    for i in range(ttime):
        result = algo.work(requests[i], i)
    if algoIndex == 1:
        for req in algo.requestState:
            if algo.requestState[req].state == 2:
                algo.requestState[req].intermediate.clearIntermediate()
    t[numOfRequestPerRound][algoIndex] = result

if __name__ == '__main__':
    global t
    topo = Topo.generate(100, 0.9, 5, 0.05, 6)
    f = open('../../plot/data/data.txt', 'w')

    numOfRequestPerRoundMax = 2

    # make copy
    algorithms = []
    algorithms.append(GreedyHopRouting(copy.deepcopy(topo)))
    algorithms.append(MyAlgorithm(copy.deepcopy(topo)))
    algorithms.append(GreedyGeographicRouting(copy.deepcopy(topo)))
    algorithms.append(OnlineAlgorithm(copy.deepcopy(topo)))
    # algorithms.append(REPS(copy.deepcopy(topo)))

    t = [[0 for _ in range(len(algorithms))] for _ in range(numOfRequestPerRoundMax + 1)]
    ttime = 200
    rtime = 10

    threads = [[] for _ in range(numOfRequestPerRoundMax + 1)]

    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        # samplesPerTime = 2 * numOfRequestPerRound
        # requests = {i : [] for i in range(ttime)}
        # for i in range(ttime):
        #     if i < rtime:
        #         for _ in range(numOfRequestPerRound):
        #             a = sample(topo.nodes, 2)
        #             requests[i].append((a[0], a[1]))

        for algoIndex in range(len(algorithms)):
            algo = copy.deepcopy(algorithms[algoIndex])
            # Job = threading.Thread(target = run, args = (algo, requests.copy(), algoIndex, numOfRequestPerRound, ttime))
            # threads[numOfRequestPerRound].append(Job)

            requests = {i : [] for i in range(ttime)}
            for i in range(ttime):
                if i < rtime:
                    for _ in range(numOfRequestPerRound):
                        a = sample([i for i in range(100)], 2)
                        requests[i].append((algo.topo.nodes[a[0]], algo.topo.nodes[a[1]]))
            
            Job = threading.Thread(target = run, args = (algo, requests, algoIndex, numOfRequestPerRound, ttime))
            threads[numOfRequestPerRound].append(Job)

    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        for algoIndex in range(len(algorithms)):
            threads[numOfRequestPerRound][algoIndex].start()


    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        for algoIndex in range(len(algorithms)):
            threads[numOfRequestPerRound][algoIndex].join()

    for numOfRequestPerRound in range(1, numOfRequestPerRoundMax + 1):
        f.write(str(numOfRequestPerRound))
        for algoIndex in range(len(algorithms)):
            f.write(' ')
            f.write(str(numOfRequestPerRound))
        f.write('\n')
    # 5XX
    f.close()
    