import multiprocessing
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
from REPS_cache import REPSCACHE
from REPS_cache2 import REPSCACHE2
from REPS_cache_preswap import REPSCACHE4
from REPS_cache4 import REPSCACHE5
from SEER_cache import SEERCACHE
from SEER_cache2 import SEERCACHE2
from SEER_cache3 import SEERCACHE3
from SEER_cache3_2 import SEERCACHE3_2
from SEE import SEE
from SEE2 import SEE2
from CachedEntanglement import CachedEntanglement
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link
from random import sample
from numpy import log as ln
import numpy as np
import random
import time

def runThread(algo, requests, algoIndex, ttime, pid, resultDict):
    for i in range(ttime):
        result = algo.work(requests[i], i)
    if algo.name == "My" or 'SEER' in algo.name:
        print('============ in runThread', algo.name)
        for req in algo.requestState:
            if algo.requestState[req].state == 2:
                algo.requestState[req].intermediate.clearIntermediate()
    resultDict[pid] = result



def Run(numOfRequestPerRound = 20, numOfNode = 50, r = 7, q = 0.8, alpha = 0.0002, SocialNetworkDensity = 0.5, rtime = 101, topo = None, FixedRequests = None , results=[]):

    if topo == None:
        topo = Topo.generate(numOfNode, q, 5, alpha, 6)
    
    topo.setQ(q)
    topo.setAlpha(alpha)

    # make copy
    algorithms = []

    # algorithms.append(MyAlgorithm(copy.deepcopy(topo)))
    # algorithms.append(SEERCACHE(copy.deepcopy(topo), param = 'ten', name='SEER2'))
    # algorithms.append(SEERCACHE2(copy.deepcopy(topo), param = 'ten', name='SEER3'))
    # algorithms.append(SEERCACHE3(copy.deepcopy(topo), param = 'ten', name='SEER4'))



    # algorithms.append(SEERCACHE3_2(copy.deepcopy(topo), param = 'ten', name='SEER4_2'))

    #with pre entanglement
    # algorithms.append(MyAlgorithm(copy.deepcopy(topo),preEnt=True))
    # algorithms.append(GreedyHopRouting(copy.deepcopy(topo)))
    # algorithms.append(GreedyGeographicRouting(copy.deepcopy(topo)))


    # algorithms.append(OnlineAlgorithm(copy.deepcopy(topo)))
    # algorithms.append(CachedEntanglement(copy.deepcopy(topo)))

    # #with pre entanglement
    # algorithms.append(CachedEntanglement(copy.deepcopy(topo),preEnt=True))
    
    algorithms.append(REPS(copy.deepcopy(topo)))
    algorithms.append(REPSCACHE(copy.deepcopy(topo),param='ten',name='REPSCACHE2'))
    algorithms.append(REPSCACHE2(copy.deepcopy(topo),param='ten',name='REPSCACHE3'))
    # algorithms.append(REPSCACHE4(copy.deepcopy(topo),param='ten',name='REPSCACHE4'))
    algorithms.append(REPSCACHE5(copy.deepcopy(topo),param='ten',name='REPSCACHE5'))

    
    # algorithms.append(SEE(copy.deepcopy(topo)))

    algorithms[0].r = r
    algorithms[0].density = SocialNetworkDensity

    times = 3
    # times = 10
    results = [[] for _ in range(len(algorithms))]
    ttime = rtime
    rtime = ttime

    resultDicts = [multiprocessing.Manager().dict() for _ in algorithms]
    jobs = []


    bias_weights = [x%10==0 for x in range(numOfNode)]
    prob = np.array(bias_weights) / np.sum(bias_weights)


    pid = 0
    for _ in range(times):
        ids = {i : [] for i in range(ttime)}
        if FixedRequests != None:
            ids = FixedRequests
        else:
            for i in range(ttime):
                if i < rtime:
                    for _ in range(numOfRequestPerRound):
                        # a = sample([i for i in range(numOfNode)], 2)

                        a = np.random.choice(len(prob), size=2, replace=False, p=prob)
                        # print('req: ' , a)
                        # for _ in range(int(random.random()*3+1)):
                        ids[i].append((a[0], a[1]))
                # print('#############################  ', len(ids[i]))
        
        for algoIndex in range(len(algorithms)):
            algo = copy.deepcopy(algorithms[algoIndex])
            requests = {i : [] for i in range(ttime)}
            for i in range(rtime):
                for (src, dst) in ids[i]:
                    # print(src, dst)
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst]))
            
            pid += 1
            job = multiprocessing.Process(target = runThread, args = (algo, requests, algoIndex, ttime, pid, resultDicts[algoIndex]))
            jobs.append(job)

    for job in jobs:
        job.start()
        time.sleep(1)

    for job in jobs:
        job.join()

    # print(resultDicts)
    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(resultDicts[algoIndex].values())


    # results[0] = result of GreedyHopRouting = a AlgorithmResult
    # results[1] = result of MyAlgorithm
    # results[2] = result of GreedyGeographicRouting
    # results[3] = result of OnlineAlgorithm
    # results[4] = result of REPS

    return results
    
def mainThreadReqPerTime(Xparam , topo, result):
    result.extend(Run(numOfRequestPerRound = Xparam, topo = copy.deepcopy(topo)))
def mainThreadNumOfNode(Xparam , result):
    result.extend(Run(numOfNode = Xparam))
def mainThreadSwapProb(Xparam , topo , result):
    result.extend(Run(q = Xparam , topo=copy.deepcopy(topo)))
def mainThreadAlpha(Xparam , topo , result):
    result.extend(Run(alpha = Xparam, topo = copy.deepcopy(topo)))
def mainThreadSwapFrac(Xparam , topo , result):
    topo.preSwapFraction = Xparam
    result.extend(Run(topo = copy.deepcopy(topo)))





if __name__ == '__main__':
    print("start Run and Generate data.txt")
    targetFilePath = "../../plot/data/"
    temp = AlgorithmResult()
    Ylabels = temp.Ylabels # Ylabels = ["algorithmRuntime", "waitingTime", "idleTime", "usedQubits", "temporaryRatio"]
    
    # numOfRequestPerRound = [1, 2, 3]
    numOfRequestPerRound = [10 , 15, 20 ,25]
    # numOfRequestPerRound = [2]
    totalRequest = [10, 20, 30, 40, 50]
    numOfNodes = [50, 100, 150, 200]
    # numOfNodes = [20]
    r = [0, 2, 4, 6, 8, 10]
    q = [0.7, 0.75, 0.8, 0.85, 0.9]
    alpha = [0.0005, 0.001, 0.0015, 0.002]
    # alpha = [0.001 , 0.0015 , 0.002 , 0.0025, 0.003 , 0.0035 ]
    SocialNetworkDensity = [0.25, 0.5, 0.75, 1]

    preSwapFraction = [0.2,  0.4,  0.6,  0.8 ,  1]
    # preSwapFraction = [0.2, 0.3]

    # mapSize = [(1, 2), (100, 100), (50, 200), (10, 1000)]

    Xlabels = ["#RequestPerRound", "totalRequest", "#nodes", "r", "swapProbability", "alpha", "SocialNetworkDensity" , "preSwapFraction"]
    Xparameters = [numOfRequestPerRound, totalRequest, numOfNodes, r, q, alpha, SocialNetworkDensity, preSwapFraction]

    topo = Topo.generate(50, 0.8, 5, 0.0002, 6)
    jobs = []

    tmp_ids = {i : [] for i in range(200)}
    for i in range(200):
        if i < 20:
            for _ in range(5):
                a = sample([i for i in range(100)], 2)
                tmp_ids[i].append((a[0], a[1]))
               

    skipXlabel = [  1, 2, 3,4 , 5, 6, 7]
    for XlabelIndex in range(len(Xlabels)):
        # continue
        Xlabel = Xlabels[XlabelIndex]
        Ydata = []
        jobs = []
        results = {Xparam : multiprocessing.Manager().list() for Xparam in Xparameters[XlabelIndex]}
        pid = 0
        if XlabelIndex in skipXlabel:
            continue
        for Xparam in Xparameters[XlabelIndex]:
            # results[Xparam] = None
            
            # check schedule
            # statusFile = open("status.txt", "w")
            # print(Xlabel + str(Xparam), file = statusFile)
            # statusFile.flush()
            # statusFile.close()
            # ------
            if XlabelIndex == 0: # #RequestPerRound
                # result =[]
                job = multiprocessing.Process(target = mainThreadReqPerTime, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)
                # result = Run(numOfRequestPerRound = Xparam, topo = copy.deepcopy(topo))
            # if XlabelIndex == 1: # totalRequest
            #     result = Run(numOfRequestPerRound = Xparam, rtime = 1, topo = copy.deepcopy(topo))
            if XlabelIndex == 2: # #nodes
                # result = Run(numOfNode = Xparam)
                job = multiprocessing.Process(target = mainThreadNumOfNode, args = (Xparam  , results[Xparam] ))
                jobs.append(job)
            # if XlabelIndex == 3: # r
            #     result = Run(r = Xparam, topo = copy.deepcopy(topo), FixedRequests = tmp_ids)
            if XlabelIndex == 4: # swapProbability
                # result = Run(q = Xparam, topo = copy.deepcopy(topo))
                job = multiprocessing.Process(target = mainThreadSwapProb, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)
            if XlabelIndex == 5: # alpha
                # result = Run(alpha = Xparam, topo = copy.deepcopy(topo))
                job = multiprocessing.Process(target = mainThreadAlpha, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)

            # if XlabelIndex == 6: # SocialNetworkDensity
            #     result = Run(SocialNetworkDensity = Xparam, topo = copy.deepcopy(topo))

            if XlabelIndex == 7: # pre swap fraction
                job = multiprocessing.Process(target = mainThreadSwapFrac, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)

            # if XlabelIndex == 7:
            #     result = Run(mapSize = Xparam)
            # Ydata.append(result)

        for job in jobs:
            job.start()
            time.sleep(1)

        for job in jobs:
            job.join()
        
        for Xparam in Xparameters[XlabelIndex]:
            result = results[Xparam]
            # print('--------------printing results ---------------')
            # print(result)
            Ydata.append(result)



        for Ylabel in Ylabels:
            filename = Xlabel + "_" + Ylabel + ".txt"
            F = open(targetFilePath + filename, "w")
            for i in range(len(Xparameters[XlabelIndex])):
                Xaxis = str(Xparameters[XlabelIndex][i])
                Yaxis = [algoResult.toDict()[Ylabel] for algoResult in Ydata[i]]
                Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
                F.write(Xaxis + Yaxis)
            F.close()

    print('-----EXIT-----')
    exit(0)
    # write remainRequestPerRound
    rtime = 101
    print('starting.. ')
    # sampleRounds = [0, 2, 4, 6, 8, 10]
    sampleRounds = [i for i in range(0 , rtime , int(rtime/5))]
    print(sampleRounds)
    results = Run(numOfRequestPerRound = 20, numOfNode=100, rtime = rtime) # algo1Result algo2Result ...
    for result in results:
        result.remainRequestPerRound.insert(0, 1)
        result.entanglementPerRound.insert(0, 1)
    
    # sampleRounds = [0, 5, 10, 15, 20, 25]

    filename = "Timeslot" + "_" + "#remainRequest" + ".txt"
    F = open(targetFilePath + filename, "w")
    for roundIndex in sampleRounds:
        Xaxis = str(roundIndex)
        Yaxis = [result.remainRequestPerRound[roundIndex] for result in results]
        Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
        F.write(Xaxis + Yaxis)
    F.close()



    # filename = "Timeslot" + "_" + "#entanglement" + ".txt"
    # F = open(targetFilePath + filename, "w")
    # for roundIndex in sampleRounds:
    #     Xaxis = str(roundIndex)
    #     Yaxis = [result.entanglementPerRound[roundIndex] for result in results]
    #     Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
    #     F.write(Xaxis + Yaxis)
    # F.close()

    print('--DONE--')

