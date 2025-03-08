import multiprocessing
from objsize import get_deep_size
import gc
import sys
sys.setrecursionlimit(2000) # Increase limit to 2000
import copy
sys.path.append("../..")
# from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
# from MyAlgorithm import MyAlgorithm
# from OnlineAlgorithm import OnlineAlgorithm
# from GreedyGeographicRouting import GreedyGeographicRouting
# from GreedyHopRouting import GreedyHopRouting
from REPS import REPS
from REPS_rep import REPSREP
# from REPS_cache import REPSCACHE
# from REPS_cache2 import REPSCACHE2
# from REPS_cache4 import REPSCACHE5
# from REPS_cache4_3 import REPSCACHE5_3
# from REPS_ent_dqrl import REPS_ENT_DQRL
# from REPS_cache_ent_dqrl import REPSCACHEENT_DQRL
# from REPS_cache_ent_dqrl_proswap import REPSCACHEENT_DQRL_PSWAP
# from SEER_cache import SEERCACHE
# from SEER_cache2 import SEERCACHE2
# from SEER_cache3 import SEERCACHE3
# from SEER_cache3_3 import SEERCACHE3_3
# from SEER_ent_dqrl import SEER_ENT_DQRL
# from SEE import SEE
# from SEE2 import SEE2
from DQRL import QuRA_DQRL
from Schedule import SCHEDULEGREEDY
from ScheduleRoute import SCHEDULEROUTEGREEDY
from ScheduleRoute_cache import SCHEDULEROUTEGREEDY_CACHE
from ScheduleRoute_cache_ps import SCHEDULEROUTEGREEDY_CACHE_PS
from Heuristic import QuRA_Heuristic
# from CachedEntanglement import CachedEntanglement
from topo.Topo import Topo


from random import sample
import numpy as np
import time
import os.path
import multiprocessing.context as ctx
ctx._force_start_method('spawn')

# sys.path.insert(0, "/home/tasdiqul/Documents/Quantum Network/Projects/QuantumRouting-python/src/rl")
sys.path.insert(0, "../../rl")

# sys.path.insert(0, "/Users/tasdiqulislam/Documents/Quantum Network/Routing/QuantumRouting-python/src/rl") #for my mac
# sys.path.insert(0, "/users/Tasdiq/QuantumRouting-python/src/rl") #for cloudlab
# from agent import Agent    #for ubuntu
# # from agent import Agent   #for mac


# from DQNAgent import DQNAgent   
from DQNAgentDist import DQNAgentDist
from DQRLAgent import ALPHA, BETA,GAMMA,LEARNING_RATE,DISCOUNT,UPDATE_TARGET_EVERY, FAILURE_REWARD,clip_value
from DQRLAgent import lr,START_EPSILON_DECAYING,SKIP_REWAD,MINIBATCH_SIZE,REPLAY_MEMORY_SIZE,DELTA
# from DQNAgentDistEnt import DQNAgentDistEnt
# from DQNAgentDistEnt_2 import DQNAgentDistEnt_2
# from DQRLAgent import DQRLAgent
# from SchedulerAgent import SchedulerAgent

run = "ALPHA = " + str(ALPHA) + " BETA = " +str(BETA) + " GAMMA = "+str(GAMMA) + " DELTA = "+ str(DELTA) + " lr "+str(LEARNING_RATE)\
+" discount "+str(DISCOUNT)+" failure reward = "+str(FAILURE_REWARD)\
+", then done implemented+ skip link  rand 3 req 5 gs ute "\
+str(UPDATE_TARGET_EVERY)+" skip for no targetpath alr= " + str(lr) + "clip_value " + str(clip_value) \
+ " START_EPSILON_DECAYING " + str(START_EPSILON_DECAYING) + "SKIP REWARD "\
+ str(SKIP_REWAD) + ' MINIBATCH_SIZE ' + str(MINIBATCH_SIZE) \
    +'REPLAY_MEMORY_SIZE' + str(REPLAY_MEMORY_SIZE)+ " reward/10 as recursive -1/e 10 -10 input without q in state+=  2 6 .8"
 
ttime = 500
ttime2 = 500
step = 500
times = 1
gridSize = 3
nodeNo = gridSize *gridSize
fixed = True

alpha_ = 0.0007
degree = 1
# numOfRequestPerRound = [1, 2, 3]
# numOfRequestPerRound = [15 , 20 , 25]
# numOfRequestPerRound = [25,30,35]
numOfRequestPerRound = [100]
totalRequest = [10, 20, 30, 40, 50]
numOfNodes = [50 , 75 , 100 ]
# numOfNodes = [20]
r = [0, 2, 4, 6, 8, 10]
q = [0.7, 0.8, 0.9]
alpha = [0.001 , 0.002 , 0.003]
fidelity = [ .5 , .6 ,  .7 , .8  ]
# alpha = [0.001 , 0.0015 , 0.002 , 0.0025, 0.003 , 0.0035 ]
SocialNetworkDensity = [0.25, 0.5, 0.75, 1]
preSwapFraction = [0.4,  0.6,  0.8 ,  1]
# preSwapFraction = [0.2, 0.3]
entanglementLifetimes = [1]
requestTimeouts = [100,200,300]
preSwapCapacity = [0.2 , 0.4, 0.5, 0.6, 0.8]
skipXlabel = [ 1,2,  3 ,4,5 , 6 ,7,8 , 9]
runLabel = [11]
Xlabels = ["#RequestPerRound", "totalRequest", "#nodes", "r", "swapProbability", "alpha", "SocialNetworkDensity" , "preSwapFraction" , 'entanglementLifetime' , 'requestTimeout' , "preSwapCapacity" , 'fidelityThreshold']
toRunLessAlgos = ['REPS','REPS_shortest','QuRA_Heuristic' ,'REPS_rep', 'REPSCACHE' , 'REPSCACHE2' , 'REPS_preswap_1hop_dqrl','QuRA_DQRL_entdqrl_greedy_only', 'RANDSCHEDULEGREEDY','RANDSCHEDULEROUTEGREEDY']


def runThread(algo, requests, algoIndex, ttime, pid, resultDict , shared_data):
    # if '_qrl' in algo.name:
    #     agent = Agent(algo , pid)
    # if '_dqrl' in algo.name:
    #     agent = DQNAgent(algo , pid)
    if '_distdqrl' in algo.name:
        agent = DQNAgentDist(algo , pid)
    # if '_entdqrl' in algo.name:
    #     algo.entAgent = DQNAgentDistEnt(algo, pid)
    # if '_2entdqrl' in algo.name:
    #     algo.entAgent = DQNAgentDistEnt_2(algo, pid)
    # algo.routingAgent = DQRLAgent(algo , 0)
    # algo.schedulerAgent = SchedulerAgent(algo , pid)
    
    timeSlot = ttime
    global ttime2
    if algo.name in toRunLessAlgos:
        timeSlot = min(ttime2,ttime)

    for i in range(timeSlot):
        if '_qrl' in algo.name or '_dqrl' in algo.name or '_distdqrl' in algo.name:
            agent.learn_and_predict()

        # print([(r[0].id, r[1].id) for r in requests[i]])
        
        result = algo.work(requests[i], i)

        if '_qrl' in algo.name or '_dqrl' in algo.name or '_distdqrl' in algo.name:
            agent.update_reward()

    if algo.name == "My" or 'SEER' in algo.name:
        print('============ in runThread', algo.name)
        for req in algo.requestState:
            if algo.requestState[req].state == 2:
                algo.requestState[req].intermediate.clearIntermediate()
    resultDict[pid] = result


    success_req = 0
    
    for i in range(timeSlot):
        success_req += result.successfulRequestPerRound[i]
    max_success = algo.name + str(len(algo.topo.nodes))+str(algo.topo.alpha)+str(algo.topo.q)+ 'max_success'

    print('====================================================')
    print('====================================================')
    print('pid: ' , pid , 'success_req: ' , success_req)
    print('pid: ' , pid , 'max_success rate : ' , shared_data[max_success] / timeSlot)
    print('====================================================')
    print('====================================================')
    
    if ('_entdqrl' in algo.name or '_2entdqrl' in algo.name ) and success_req > shared_data[max_success]:
        print('going to save the ent model ' , success_req)
        # algo.entAgent.save_model()
        if hasattr(algo , 'routingAgent' ) and algo.routingAgent is not None:
            try:
                algo.routingAgent.save_model()
                print('going to save the routing model ' , success_req)
            except:
                print('couldnt save model')

        shared_data[max_success] = success_req



def Run(numOfRequestPerRound = 5, numOfNode = 0, r = 7, q = 1, alpha = alpha_, SocialNetworkDensity = 0.5, rtime = ttime, topo = None, FixedRequests = None , results=[]):

    if topo == None:
        topo = Topo.generate(numOfNode, q, 5, alpha, 6)
    numOfNode = len(topo.nodes)
    
    topo.setQ(q)
    topo.setAlpha(alpha)
    topo.setNumOfRequestPerRound(numOfRequestPerRound)


    # topo.setQ(1)
    # topo.setAlpha(0)

    # make copy
    algorithms = []

    # algorithms.append(MyAlgorithm(copy.deepcopy(topo)))
    # algorithms.append(SEERCACHE(copy.deepcopy(topo), param = 'ten', name='SEERCACHE'))

    # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_1hop'))
    # # # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_1hop_qrl'))
    # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_1hop_dqrl'))
    # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_1hop_distdqrl'))

    # # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_multihop'))
    # # # # # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_multihop_qrl'))
    # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_multihop_dqrl'))
    # algorithms.append(SEERCACHE3_3(copy.deepcopy(topo), param = 'ten', name='SEER_preswap_multihop_distdqrl'))

    # algorithms.append(MyAlgorithm(copy.deepcopy(topo) , name='SEER_entdqrl'))

    #with pre entanglement
    # algorithms.append(MyAlgorithm(copy.deepcopy(topo),preEnt=True))
    # algorithms.append(GreedyHopRouting(copy.deepcopy(topo)))
    # algorithms.append(GreedyGeographicRouting(copy.deepcopy(topo)))


    # algorithms.append(OnlineAlgorithm(copy.deepcopy(topo)))
    # algorithms.append(CachedEntanglement(copy.deepcopy(topo)))

    # #with pre entanglement
    # algorithms.append(CachedEntanglement(copy.deepcopy(topo),preEnt=True))
    

    # algorithms.append(REPS(copy.deepcopy(topo) , name = 'REPS'))
    algorithms.append(REPSREP(copy.deepcopy(topo) , name = 'REPS_rep'))
    # algorithms.append(REPSREP(copy.deepcopy(topo) , name = 'REPS_shortest'))
    # algorithms.append(REPS(copy.deepcopy(topo) , name = 'REPS', param = 'reps_ten'))
    # algorithms.append(REPSCACHE(copy.deepcopy(topo),param='ten',name='REPSCACHE2'))

    # # # # # algorithms.append(REPSCACHE2(copy.deepcopy(topo),param='ten',name='REPSCACHE3'))
    # # # # # # # # algorithms.append(REPSCACHE5(copy.deepcopy(topo),param='ten',name='REPSCACHE5'))

    # # algorithms.append(REPSCACHE5_3(copy.deepcopy(topo),param='ten',name='REPS_preswap_1hop'))
    # # algorithms.append(REPSCACHE5_3(copy.deepcopy(topo),param='ten',name='REPS_preswap_1hop_qrl'))
    # algorithms.append(REPSCACHE5_3(copy.deepcopy(topo),param='ten',name='REPS_preswap_1hop_dqrl'))
    
    # # algorithms.append(REPSCACHE5_3(copy.deepcopy(topo),param='ten',name='REPSCACHE5_preswap_multihop'))
    # # algorithms.append(REPSCACHE5_3(copy.deepcopy(topo),param='ten',name='REPSCACHE5_preswap_multihop_qrl'))
    # # algorithms.append(REPSCACHE5_3(copy.deepcopy(topo),param='ten',name='REPSCACHE5_preswap_multihop_dqrl')) #working
    
    
    # algorithms.append(REPS_ENT_DQRL(copy.deepcopy(topo) , name='REPS_entdqrl'))
    # # algorithms.append(REPS_ENT_DQRL(copy.deepcopy(topo) , name='REPS_entdqrl_no_repeat'))
    # # algorithms.append(REPS_ENT_DQRL(copy.deepcopy(topo) , name='REPS_2entdqrl'))
    # # algorithms.append(REPS_ENT_DQRL(copy.deepcopy(topo) , name='REPS_2entdqrl_no_repeat'))

    # algorithms.append(REPSCACHEENT_DQRL(copy.deepcopy(topo),param='ten',name='REPSCACHE_entdqrl'))
    
    # # algorithms.append(REPSCACHEENT_DQRL(copy.deepcopy(topo),param='ten',name='REPSCACHE_entdqrl_no_repeat'))
    # # algorithms.append(REPSCACHEENT_DQRL(copy.deepcopy(topo),param='ten',name='REPSCACHE_2entdqrl'))
    # # algorithms.append(REPSCACHEENT_DQRL(copy.deepcopy(topo),param='ten',name='REPSCACHE_2entdqrl_no_repeat'))

    # algorithms.append(REPSCACHEENT_DQRL_PSWAP(copy.deepcopy(topo),param='ten',name='REPSCACHE_DQRL_PSWAP_entdqrl_1hop_distdqrl'))
    
    # # algorithms.append(REPSCACHEENT_DQRL_PSWAP(copy.deepcopy(topo),param='ten',name='REPSCACHE_DQRL_PSWAP_entdqrl_no_repeat_1hop_distdqrl'))
    # # algorithms.append(REPSCACHEENT_DQRL_PSWAP(copy.deepcopy(topo),param='ten',name='REPSCACHE_DQRL_PSWAP_2entdqrl_1hop_distdqrl'))
    # # algorithms.append(REPSCACHEENT_DQRL_PSWAP(copy.deepcopy(topo),param='ten',name='REPSCACHE_DQRL_PSWAP_2entdqrl_no_repeat_1hop_distdqrl'))

    
    # algorithms.append(SEE(copy.deepcopy(topo)))

    algorithms.append(QuRA_DQRL(copy.deepcopy(topo) , name = 'QuRA_DQRL_entdqrl'))
    algorithms.append(QuRA_DQRL(copy.deepcopy(topo) , name = 'QuRA_DQRL_entdqrl_greedy_only' , param = 'greedy_only'))
   
    # algorithms.append(QuRA_Heuristic(copy.deepcopy(topo) , name = 'QuRA_Heuristic'))

   
    print('======================before append', Topo.print_memory_usage())
   
    # algorithms.append(SCHEDULEGREEDY(copy.deepcopy(topo) , name = 'SCHEDULEGREEDY'))
    # algorithms.append(SCHEDULEGREEDY(copy.deepcopy(topo) , name = 'SCHEDULEGREEDY_prob'))
    # algorithms.append(SCHEDULEGREEDY(copy.deepcopy(topo) , name = 'RANDSCHEDULEGREEDY'))






    # algorithms.append(SCHEDULEROUTEGREEDY(copy.deepcopy(topo) , name = 'SCHEDULEROUTEGREEDY'))
    # algorithms.append(SCHEDULEROUTEGREEDY(copy.deepcopy(topo) , name = 'RANDSCHEDULEROUTEGREEDY'))

    # algorithms.append(SCHEDULEROUTEGREEDY_CACHE(copy.deepcopy(topo) , name = 'SCHEDULEROUTEGREEDY_CACHE' , param='ten'))
    # algorithms.append(SCHEDULEROUTEGREEDY_CACHE(copy.deepcopy(topo) , name = 'RANDSCHEDULEROUTEGREEDY_CACHE' , param='ten'))
    
    # algorithms.append(SCHEDULEROUTEGREEDY_CACHE(copy.deepcopy(topo) , name = 'SCHEDULEROUTEGREEDY_CACHE' , param='ten'))
    # algorithms.append(SCHEDULEROUTEGREEDY_CACHE_PS(copy.deepcopy(topo) , name = 'RANDSCHEDULEROUTEGREEDY_CACHE_preswap_multihop_distdqrl' , param='ten'))
    
    gc.collect()
    print('======================after append', Topo.print_memory_usage())
    


    algorithms[0].r = r
    algorithms[0].density = SocialNetworkDensity

    global times
    # times = 10
    results = [[] for _ in range(len(algorithms))]
    ttime = rtime
    rtime = ttime

    resultDicts = [multiprocessing.Manager().dict() for _ in algorithms]
    shared_data = multiprocessing.Manager().dict()
   
    for algo in algorithms:
        max_success = algo.name + str(len(algo.topo.nodes))+str(algo.topo.alpha)+str(algo.topo.q)+ 'max_success'
        shared_data[max_success] = 0
    
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
                    if fixed:
                        ids[i] = topo.generateRequest(numOfRequestPerRound)
                    else:
                        for _ in range(numOfRequestPerRound):

                            while True:
                                a = sample([i for i in range(numOfNode)], 2)
                                if (a[0], a[1]) not in ids[i]:
                                    break


                            # a = [2 , 25]

                            # a = np.random.choice(len(prob), size=2, replace=False, p=prob)
                            # print('req: ' , a)
                            # for _ in range(int(random.random()*3+1)):
                            ids[i].append((a[0], a[1]))
        print('##############going to append jobs ###############  ')
        # print('----------size(ids)----------------', get_deep_size(ids)/1000000)
        # print('----------size(algorithms)----------------', get_deep_size(algorithms)/1000000)

        
        for algoIndex in range(len(algorithms)):
            algo = copy.deepcopy(algorithms[algoIndex])
            requests = {i : [] for i in range(ttime)}
            for i in range(rtime):
                for (src, dst) in ids[i]:
                    # print(src, dst)
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst]))
            
            pid += 1
            job = multiprocessing.Process(target = runThread, args = (algo, requests, algoIndex, ttime, pid, resultDicts[algoIndex] , shared_data))
            jobs.append(job)

    for job in jobs:
        job.start()
        # time.sleep(1)

    for job in jobs:
        job.join()

    # print(resultDicts)
    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(resultDicts[algoIndex].values(), numOfRequestPerRound , algorithms[0].topo)


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
def mainThreadEntanglementLifetime(Xparam , topo , result):
    topo.entanglementLifetime = Xparam
    result.extend(Run(topo = copy.deepcopy(topo)))
def mainThreadRequestTimeout(Xparam , topo , result):
    topo.requestTimeout = Xparam
    result.extend(Run(topo = copy.deepcopy(topo)))
def mainThreadPreSwapCapacity(Xparam , topo , result):
    topo.preswap_capacity = Xparam
    result.extend(Run(topo = copy.deepcopy(topo)))
def mainThreadFidelityThreshold(Xparam , topo , result):
    topo.fidelity_threshold = Xparam
    result.extend(Run(topo = copy.deepcopy(topo)))





if __name__ == '__main__':
    print("start Run and Generate data.txt")
    print("runrunrunrun " , run)
    t1 = time.time()
    targetFilePath = "../../plot/data/"
    temp = AlgorithmResult()
    Ylabels = temp.Ylabels # Ylabels = ["algorithmRuntime", "waitingTime", "idleTime", "usedQubits", "temporaryRatio"]
    


    # mapSize = [(1, 2), (100, 100), (50, 200), (10, 1000)]

    Xparameters = [numOfRequestPerRound, totalRequest, numOfNodes, r, q, alpha, SocialNetworkDensity, preSwapFraction, entanglementLifetimes , requestTimeouts , preSwapCapacity , fidelity]

    print('--------calling topo.generate() ---------------')
    topo = Topo.generate(nodeNo, 0.9, 5,alpha_, degree, gridSize=gridSize)
    jobs = []

    tmp_ids = {i : [] for i in range(200)}
    for i in range(200):
        if i < 20:
            for _ in range(5):
                a = sample([i for i in range(100)], 2)
                tmp_ids[i].append((a[0], a[1]))
               
    output = ''
    for XlabelIndex in range(len(Xlabels)):
        # continue
        Xlabel = Xlabels[XlabelIndex]
        Ydata = []
        jobs = []
        results = {Xparam : multiprocessing.Manager().list() for Xparam in Xparameters[XlabelIndex]}
        pid = 0
        # if XlabelIndex in skipXlabel:
        #     continue
        if XlabelIndex not in runLabel:
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
            
            if XlabelIndex == 8: # entanglement lifetime
                job = multiprocessing.Process(target = mainThreadEntanglementLifetime, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)
            if XlabelIndex == 9: # request timeout
                job = multiprocessing.Process(target = mainThreadRequestTimeout, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)
            if XlabelIndex == 10: # pre swap capacity
                job = multiprocessing.Process(target = mainThreadPreSwapCapacity, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)
            if XlabelIndex == 11: # pre swap capacity
                job = multiprocessing.Process(target = mainThreadFidelityThreshold, args = (Xparam , topo , results[Xparam] ))
                jobs.append(job)

            # if XlabelIndex == 7:
            #     result = Run(mapSize = Xparam)
            # Ydata.append(result)

        for job in jobs:
            job.start()
            # time.sleep(1)

        for job in jobs:
            job.join()
        
        for Xparam in Xparameters[XlabelIndex]:
            result = results[Xparam]
            # print('--------------printing results ---------------')
            # print(result)
            Ydata.append(result)


        print(run)
        for i in range(len(Xparameters[XlabelIndex])):
            Xparam = Xparameters[XlabelIndex][i]
            filename = "Timeslot" + "_" + "#successRequest"+ str(Xparam) + ".txt"
            sampleRounds = [s for s in range(0 , ttime , step)]
            print(filename)
            output += filename + '\n'


            F = open(targetFilePath + filename, "w")
            for roundIndex in sampleRounds:
                Xaxis = str(roundIndex)
                # Yaxis1 = [result.successfulRequestPerRound[roundIndex] for result in Ydata[0]]
                Yaxis = []
                # try:
                #     Yaxis = [sum(result.successfulRequestPerRound[roundIndex:roundIndex+step])/step for result in Ydata[0]]
                # except:
                for result in Ydata[i]:
                    try:
                        Yaxis.append(sum(result.successfulRequestPerRound[roundIndex:roundIndex+step])/step)
                    except:
                        Yaxis.append(0)
                # print('Yaxis ' , roundIndex , Yaxis1)
                Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
                print(Xaxis + Yaxis.replace( "\n" , ""))
                output += Xaxis + Yaxis + '\n'

                F.write(Xaxis + Yaxis)
            F.close()

        filename = "Timeslot" + "_" + "reward" + ".txt"
        sampleRounds = [i for i in range(0 , ttime , step)]
        print(filename)
        output += filename + '\n'


        F = open(targetFilePath + filename, "w")
        for roundIndex in sampleRounds:
            Xaxis = str(roundIndex)
            Yaxis = []

            for result in Ydata[0]:
                try:
                    Yaxis.append(sum(result.rewardPerRound[roundIndex:roundIndex+step])/step)
                except:
                    Yaxis.append(0)
            # print('Yaxis ' , roundIndex , Yaxis1)
            Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
            print(Xaxis + Yaxis.replace( "\n" , ""))
            output += Xaxis + Yaxis + '\n'

            F.write(Xaxis + Yaxis)
        F.close()

        for i in range(len(Xparameters[XlabelIndex])):
            Xparam = Xparameters[XlabelIndex][i]
            filename = "Timeslot" + "_" + "fidelity"+ str(Xparam) + ".txt"
            sampleRounds = [s for s in range(0 , ttime , step)]
            print(filename)
            output += filename + '\n'
            F = open(targetFilePath + filename, "w")
            for roundIndex in sampleRounds:
                Xaxis = str(roundIndex)
                # Yaxis1 = [result.successfulRequestPerRound[roundIndex] for result in Ydata[0]]
                Yaxis = []
                # try:
                #     Yaxis = [sum(result.successfulRequestPerRound[roundIndex:roundIndex+step])/step for result in Ydata[0]]
                # except:
                for result in Ydata[i]:
                    try:
                        Yaxis.append(sum(result.fidelityPerRound[roundIndex:roundIndex+step])/step)
                    except:
                        Yaxis.append(0)
                # print('Yaxis ' , roundIndex , Yaxis1)
                Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
                print(Xaxis + Yaxis.replace( "\n" , ""))
                output += Xaxis + Yaxis + '\n'

                F.write(Xaxis + Yaxis)
            F.close()

        for Ylabel in Ylabels:
            filename = Xlabel + "_" + Ylabel + ".txt"
            print(filename)
            output += filename + '\n'


            if os.path.isfile(targetFilePath + filename):
                F = open(targetFilePath + filename, "w")
            else:
                F = open(targetFilePath + filename, "a")
            for i in range(len(Xparameters[XlabelIndex])):
                Xaxis = str(Xparameters[XlabelIndex][i])
                Yaxis = [algoResult.toDict()[Ylabel] for algoResult in Ydata[i]]
                Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
                print(Xaxis + Yaxis.replace( "\n" , ""))
                output += Xaxis + Yaxis + '\n'

                F.write(Xaxis + Yaxis)
            F.close()

    t2 = time.time()
    output += run
    print(output)
    print('-----EXIT----- total time taken: ' , (t2-t1)/3600 , ' hours')

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

