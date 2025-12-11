from multiprocessing import Lock, Manager, shared_memory
import pickle
import copy
import sys
import math
import random
from queue import PriorityQueue
import uuid

import psutil 
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 

from numpy import log as ln
from random import sample
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from topo.mp_helper import executor as executor2,train_executor , mpredis,update_shared_topo, route_schedule_single2 , qManager, lock1, agent_lock,reward_lock, node_locks
from multiprocessing.managers import BaseManager
import dill
import requests
import httpx
import asyncio
from objsize import get_deep_size


# executor2 = ProcessPoolExecutor(max_workers=8)  # Create at the top level




# ctx._force_start_method('spawn')

sys.path.insert(0, "../../rl")
active_futures = set()
lock_manager = None
# lock = Lock()
# lock1 = Lock()


class QuRA_DQRL_DIST(AlgorithmBase):
    def __init__(self, topo,param=None, name=''):
        super().__init__(topo)
        self.name = name
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        # self.entAgent = DQNAgentDistEnt(self, 0)
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        self.hopCountThreshold = 25
        self.requestState = []
        self.optPaths = {}
        # self.pool = None
        self.w1 = 1
        self.w2 = 1 - self.w1
        self.maxTry = 2
        self.executor = None
        self.tst = [] 
        self.shared_memories = []
        self.executor_stats = []





    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        return (varName + str(parName)).replace(' ', '').replace(',', '][')
    def cleanup_shared_memory(self):
        print('cleaning up shared memory...')
        print('number of shared memories to clean:', len(self.shared_memories))
        try:
            for shm in self.shared_memories:
                try:
                    size = shm.size
                    size_mb = size / 1024 / 1024
                    print(f"******************************Cleaning shared memory: Name={shm.name}, Size={size_mb:.2f} MB")
                    shm.close()
                except Exception as e:
                    print("Error closing shared memory:", e)
                try:
                    shm.unlink()
                except Exception as e:
                    print("Error unlinking shared memory:", e)
        except Exception as e:
            print("Error cleaning up shared memory:", e)
        finally:
            self.shared_memories = []
    def printResult(self):
        self.topo.clearAllEntanglements()
        self.result.waitingTime = self.totalWaitingTime / self.totalRequest
        self.result.usedQubits = self.totalUsedQubits / self.totalRequest
        
        # self.result.remainRequestPerRound.append(len(self.requests) / self.totalRequest)
        self.result.remainRequestPerRound.append(len(self.requests))

        self.cleanup_shared_memory()
        
        print("[REPS] total time:", self.result.waitingTime)
        print("[REPS] remain request:", len(self.requests))
        print("[REPS] current Timeslot:", self.timeSlot)



        print('[REPS] idle time:', self.result.idleTime)
        print('[' , self.name, '] :' , self.timeSlot, ' total successful request::', self.result.successfulRequest)
        print('[' , self.name, '] :' , self.timeSlot, ' average path len        ::', self.result.pathlen/(self.result.successfulRequest+ 1))
        print('[' , self.name, '] :' , self.timeSlot, ' total path      ::', self.result.totalPath)

        print('[REPS] remainRequestPerRound:', self.result.remainRequestPerRound[-1])
        print('[REPS] avg usedQubits:', self.result.usedQubits)

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
            # print('addnewsdpair ' , len(self.requests) , self.timeSlot)

        self.srcDstPairs = []
        self.requestState = []
        index = 0
        for request in self.requests:
            src = request[0]
            dst = request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))
            self.requestState.append([src,dst , src.id , tuple([src.id]) , index , False])
            index += 1

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime += len(self.requests)
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            # self.PFT() # compute (self.ti, self.fi)
            self.randPFT()
            # self.entAgent.learn_and_predict()
        # print('[REPS] p2 end')
    
    def randPFT(self):
        assignable = True
        while assignable:
            assignable = False
            for link in self.topo.links:
                if link.assignable():
                    assignable = True
                    if np.random.random() > 0.5:
                    
                        link.assignQubits()
                        self.totalUsedQubits += 2
    
   
    def prep4(self):
        if len(self.srcDstPairs) > 0:
            self.Pi = {SDpair : [] for SDpair in self.srcDstPairs}

            for i in range(4):
                self.EPS()
                s = self.ELS(self.Pi)
            
                print('=====---=====prep4 ----=====----===== ' , self.timeSlot , i , s)
                if not s:
                    break

            print('=====---=====prep4 final ----=====----===== ' , self.timeSlot , i , s)
            for sd in self.Pi:
                for path in self.Pi[sd]:
                    print([n.id for n in path])
        
    def get_ent_graph_matrix(self):
        # n = len(self.topo.nodes)
        n = 100
        matrix = np.zeros((n,n))
        # node_index = {self.topo.nodes[i].id : i for i in range(n)}
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                i = link.n1.id
                j = link.n2.id
                matrix[i][j] += 1
                matrix[j][i] += 1
        return matrix
    def get_ent_graph_matrix_info(self):
        matrix = self.get_ent_graph_matrix()
        matrix *= 2
        shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
        shared_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        shared_matrix[:] = matrix[:]
        node_matrix_info = (shm.name, matrix.shape, matrix.dtype)
        self.shared_memories.append(shm)  # keep reference

        return node_matrix_info
    def dist_matrix(self):
        # n = len(self.topo.nodes)
        n = 100
        matrix = np.zeros((n,n))
        # node_index = {self.topo.nodes[i].id : i for i in range(n)}
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                i = link.n1.id
                j = link.n2.id
                matrix[i][j] = link.fidelity
                matrix[j][i] = link.fidelity

        return matrix
    def q_matrix(self):
        # n = len(self.topo.nodes)
        n = 100
        matrix = np.zeros((n))
        # node_index = {self.topo.nodes[i].id : i for i in range(n)}
        for node in self.topo.nodes:
            i = node.id
            matrix[i] = node.q
        # print('q_matrix ' , matrix)
        return matrix
    
    def req_matrix(self):
        matrix = []
        paths = []
        n = 100
        for req in self.requestState:
            r = [0 for _ in range(n)]
            r[0] = req[0].id
            r[1] = req[1].id
            r[2] = req[2]
            path = [0 for _ in range(n)]
            path[req[2]] = 1
        
            r[4] = req[4]
            r[5] = req[5]
            matrix.append(r)
            paths.append(path)
        matrix.extend(paths)

        matrix = np.array(matrix)
        return matrix
    def req_matrix_info(self):
        matrix = self.req_matrix()
        shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
        shared_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        shared_matrix[:] = matrix[:]
        node_matrix_info = (shm.name, matrix.shape, matrix.dtype)
        self.shared_memories.append(shm)  # keep reference

        return node_matrix_info

    def p4(self):
        p_time = 0
        global executor2
        global lock_manager
        global node_locks
        if lock_manager is None:
            lock_manager = Manager()
        for i in range(100):
            if i not in node_locks:
                node_locks[i] = lock_manager.Lock()
        print('start p4 ' , self.name)
        # self.prep4()

        if len(self.srcDstPairs) > 0:
            # self.EPS()
            # self.ELS()
            t = time.time()

            if 'greedy_only' in  self.name:
                self.route_seq()
                p_time += time.time()-t
            else:
                # self.route()
                # with ThreadPoolExecutor(max_workers=8) as executor:
                # with ProcessPoolExecutor(max_workers=8) as executor:
                # if not executor2:
                #     executor2 = ProcessPoolExecutor(max_workers=8)
                # with Manager() as manager:
                #     shared_nodes = manager.dict({node.id: {"remainingQubits": node.remainingQubits} for node in self.topo.nodes})
                node_matrix_info = self.get_ent_graph_matrix_info()
                req_matrix_info = self.req_matrix_info()
                dist_matrix = self.dist_matrix()
                q_matrix = self.q_matrix()

            
                args = [( node_matrix_info ,req_matrix_info, dist_matrix,q_matrix , reqState,node_locks) for reqState in self.requestState]
                # self.topo.tst = Manager().list()
                # for _ in range(10):
                #     print('going to map route_schedule_single with args2:' , len(args), len(args[0]))
                #     # route_schedule_single2(args[0])
                # for link in self.topo.links:
                #     mpredis.set('link_' + link.id, dill.dumps(link))
                # mpredis.set("shared_nodes", dill.dumps(self.topo.nodes))
                # for node in self.topo.nodes:
                #     mpredis.set("node_" + str(node.id), dill.dumps(node))
                # mpredis.set("routing_agent", dill.dumps(self.routingAgent))
                # mpredis.set("reward_routing", dill.dumps(self.topo.reward_routing))

                # print('going to map route_schedule_single with args:' )
                # print(f"\n🔍 [-----------BEFORE executor2.map] Checking executor memory...")
                # before_map = self.check_executor_memory()
                
                results = list(executor2.map(self.route_parallel, args))
                print('results from map ')
                print('got results with conflicts' , sum([r[0] for r in results]))
                node_matrix = self.get_ent_graph_matrix()
                print('node_matrix after map ')
                try:
                    successReq = self.resolve_conflict( results , q_matrix , node_matrix.copy())
                    print('got results after conflicts' , successReq)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                print('results after resolve conflict ')
                # print('results ' , results, sum([r for r in results]))
            
                # print(f"\n🔍 [->->->->-AFTER executor2.map] Checking executor memory...")
                # after_map = self.check_executor_memory()
        
                # # Compare
                # if after_map and before_map:
                #     delta = after_map['workers'] - before_map['workers']
                #     print(f"\n📊 Worker memory delta: {delta:+.2f} MB")
                #     if delta > 50:
                #         print(f"⚠️  !!!!!!!!!!!!WARNING: Workers consumed {delta:.2f} MB during this map operation!")
    
                # self.topo.reward_routing = dill.loads(mpredis.get("reward_routing"))
                # successReq = sum([r[0] for r in results])
                # print('successReq ' , successReq)
                actions = []
                
                req_matrix = self.req_matrix()
                # dist_matrix = self.dist_matrix()
                for r in results:
                    for actionss in r[1]:
                        # actionss.extend([node_matrix, req_matrix, dist_matrix])
                        actions.append(actionss)
                    # print('r[1] ' , r[1])
                    # actions.append(r[1])
                # print('total actionss ' , len(actions))
                ta = time.time()

                # try:
                #     # for  index ,current_node_id,  next_node_id  , current_state  , done_episode,reward in actions:
                #     #     self.routingAgent.update_action( index ,current_node_id,  next_node_id  , current_state  , done_episode, self.timeSlot,reward , node_matrix,req_matrix,dist_matrix)
                #     # for action in actions:
                #     #     self.call_update_action_batch([action])
                #     # self.call_update_action_batch(actions[:min(5, len(actions))])
                #     self.call_update_action_batch(actions)
                # except Exception as e:
                #     import traceback
                #     traceback.print_exc()
                # print('time for update action ======== ' , time.time() - ta)
                # print('total actionss after update ' , len(actions))
                self.result.successfulRequestPerRound.append(successReq)
                self.result.entanglementPerRound.append(successReq)
                self.result.fidelityPerRound.append(0)

                self.result.successfulRequest += successReq
                # p_time += self.route_schedule_seq()
            # self.route_seq()
            # self.route_schedule_seq()
            print('*=========time for route schedule ======== ' , time.time() - t)
        t = time.time()
        self.filterReqeuest()

        # print('[REPS] p4 end') 
        self.printResult()
        # self.entAgent.update_reward()
        reward = 0
        if not 'greedy_only' in self.name:
            if self.timeSlot < 100000:
                t = time.time()
                
                print('going to call update_reward with ')
                try:
                    self.run_async_in_thread(self.call_update_reward(
                        successful_requests=self.result.successfulRequestPerRound[-1],
                        timeSlot=self.timeSlot,
                        actions=actions
                    ))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
         
                reward = 0
                print('time for update_reward ======== ' , time.time() - t)

            a = 10
        self.result.rewardPerRound.append(reward)
        
        # print(f"\n🔍 [AFTER p4] Checking executor memory...")
        # self.check_executor_memory()
        # if self.timeSlot % 10 == 0:
        #     self.print_executor_memory_trend()
        
        p_time += time.time()-t
        self.result.p_time = p_time
        print('self.w1 w2 ' , self.w1 , self.w2)

        return self.result
    def get_action_ILP(self):
        req_index = [req[4] for req in self.requestState if not req[[5]]]
        req = self.requestState[random.choice(req_index)]
        next_node  = self.Pi[req][0][1]
        self.Pi[req][0].pop(0)
        
    

    def getConflicts(self, current_node , next_node , selectedEdgesDict):
        conflicts = []
        for req in selectedEdgesDict:
            edges = selectedEdgesDict[req]
            if (current_node, next_node) in edges:
                conflicts.append((req , current_node, next_node))
            elif (next_node, current_node) in edges:
                conflicts.append((req , next_node, current_node))
        
        return conflicts
    def resolve_conflict(self , results, q_matrix=None, node_matrix=None):
        path_infos = []
        for r in results:
            if r[0]:
                path_infos.append((r[2],r[3]))  # (index, path)
        print('path_infos before sort == ' , len(path_infos))

        path_infos.sort(key=lambda x: (len(x[1])) , reverse=True)
        success_req = 0
        for path_info in path_infos:
            index = path_info[0]
            req = self.requestState[index]

            path = path_info[1]
            swappSuccess = True
                
            # print('going to swap for ' , (src.id , dst.id ))
            for i in range(1 , len(path)-1):
                swapped = False

                if node_matrix[path[i-1]][path[i]] >= 1:
                    # print('qmatrix ' , q_matrix)
                    if random.random() <= q_matrix[path[i]]:
                        swapped = True
                if not swapped:
                    failed_swap = True
                    swappSuccess = False
                    # print('================failed swap==================')
                    break
            

                        


            if  swappSuccess:
                # print('going to find path for:', (src.id , dst.id ))

                for i in range(1 , len(path)):
                    node_matrix[path[i-1]][path[i]] -= 1
                    node_matrix[path[i]][path[i-1]] -= 1
                t2 = time.time()
                for req in self.requests:
                    src = req[0]
                    dst = req[1]
                    if (src, dst) == (req[0], req[1]):
                        # print('[REPS] finish time:', self.timeSlot - request[2])
                        self.requests.remove(req)
                        break
                success_req += 1


        return success_req
            



    def route_parallel(self, args):
        try:
            return self.route_schedule_single(args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (0, [])
    def check_executor_memory(self):
        """Check memory of executor2 and train_executor workers"""
        
        print("\n" + "="*80)
        print(f"EXECUTOR MEMORY CHECK @ timeslot {self.timeSlot}")
        print("="*80)
        
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        
        # Categorize children
        worker_processes = []
        manager_processes = []
        other_processes = []
        
        for child in children:
            try:
                cmdline = ' '.join(child.cmdline())
                # print('cmdline ' , cmdline)
                
                if 'multiprocessing.spawn' in cmdline or 'worker' in cmdline.lower():
                    worker_processes.append(child)
                elif 'semaphore_tracker' in cmdline or 'resource_tracker' in cmdline:
                    manager_processes.append(child)
                else:
                    other_processes.append(child)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Calculate memory
        worker_memory = sum(w.memory_info().rss for w in worker_processes) / 1024 / 1024
        manager_memory = sum(m.memory_info().rss for m in manager_processes) / 1024 / 1024
        other_memory = sum(o.memory_info().rss for o in other_processes) / 1024 / 1024
        parent_memory = parent.memory_info().rss / 1024 / 1024
        
        print(f"\nProcess Breakdown:")
        print(f"  Parent process:      {parent_memory:8.2f} MB")
        print(f"  Worker processes:    {len(worker_processes):3d} processes, {worker_memory:8.2f} MB")
        print(f"  Manager processes:   {len(manager_processes):3d} processes, {manager_memory:8.2f} MB")
        print(f"  Other processes:     {len(other_processes):3d} processes, {other_memory:8.2f} MB")
        print(f"  {'─'*40}")
        print(f"  Total:               {parent_memory + worker_memory + manager_memory + other_memory:8.2f} MB")
        
        # Detailed worker info
        if worker_processes:
            print(f"\nWorker Details:")
            print(f"  {'PID':<8} {'Status':<12} {'Memory (MB)':<12} {'CPU %':<10}")
            print(f"  {'-'*50}")
            
            for worker in sorted(worker_processes, 
                                key=lambda w: w.memory_info().rss, 
                                reverse=True)[:10]:  # Top 10
                try:
                    mem = worker.memory_info().rss / 1024 / 1024
                    cpu = worker.cpu_percent(interval=0.1)
                    status = worker.status()
                    
                    print(f"  {worker.pid:<8} {status:<12} {mem:<12.2f} {cpu:<10.1f}")
                except:
                    pass
        
        print("="*80)
        
        # Store stats
        stats = {
            'timeslot': self.timeSlot,
            'parent': parent_memory,
            'workers': worker_memory,
            'num_workers': len(worker_processes),
            'managers': manager_memory,
            'total': parent_memory + worker_memory + manager_memory + other_memory
        }
        self.executor_stats.append(stats)
        
        return stats
    
    def print_executor_memory_trend(self):
        """Print executor memory usage trend"""
        
        if len(self.executor_stats) < 2:
            print("Not enough data for trend analysis")
            return
        
        print("\n" + "="*80)
        print("EXECUTOR MEMORY TREND")
        print("="*80)
        
        print(f"{'Timeslot':<12} {'Parent':<10} {'Workers':<10} {'# Workers':<12} {'Total':<10}")
        print("-"*80)
        
        for stat in self.executor_stats[-10:]:  # Last 10
            print(f"{stat['timeslot']:<12} {stat['parent']:<10.2f} "
                  f"{stat['workers']:<10.2f} {stat['num_workers']:<12} "
                  f"{stat['total']:<10.2f}")
        
        # Analysis
        first = self.executor_stats[0]
        last = self.executor_stats[-1]
        
        worker_growth = last['workers'] - first['workers']
        total_growth = last['total'] - first['total']
        
        print("-"*80)
        print(f"\nGrowth Analysis:")
        print(f"  Worker memory growth:  {worker_growth:+.2f} MB")
        print(f"  Total memory growth:   {total_growth:+.2f} MB")
        print(f"  Worker count change:   {last['num_workers'] - first['num_workers']:+d}")
        
        if worker_growth > 100:
            print(f"  ⚠️  WARNING: Worker memory grew by {worker_growth:.2f} MB!")
        
        if last['num_workers'] > first['num_workers']:
            print(f"  ⚠️  WARNING: Worker count increased (possible leak)!")
        
        print("="*80)
    def run_async_in_thread(self , coro):
        global train_executor
        global active_futures
        def target():
            try:
                asyncio.run(coro)
            except Exception as e:
                import traceback
                traceback.print_exc()
        print('in run async in thread ' , len(active_futures))
        if len(active_futures) >= 10:

            # Wait for at least one to finish before submitting new one
            print('Waiting for an active future to complete.......................................')
            done, pending = wait(active_futures, return_when=FIRST_COMPLETED)

            active_futures = {f for f in active_futures if not f.done()}
            print('in run async in thread after waiting' , len(active_futures))




        try:

            future = train_executor.submit(target)
            # future.result()  # Wait for completion
            # results = list(train_executor.map(target))

            # future.join()
            active_futures.add(future)
            print('Submitted a new background task. Active tasks:', len(active_futures))
            # target()

        except Exception as e:
            import traceback
            traceback.print_exc()

    async def call_update_reward(self, successful_requests: int, timeSlot: int, actions : list):
        t = time.time()
        print('~~~~~~~~~~~in update_reward with ', timeSlot)
        mpredis.set(f"batch_{timeSlot}", pickle.dumps(actions) , ex=300)  # expire in 5 minutes
        print('**set to redis time ' , time.time() - t)
        actionIds = []
        for i in range(len(actions)):
            actionId = str(uuid.uuid4())
            mpredis.set(f"action_{actionId}", pickle.dumps(actions[i]) , ex=300)  # expire in 5 minutes
            actionIds.append(actionId)

        t = time.time()
        url = "http://127.0.0.1:8000/update_reward"

        batch_json = []
        # for param in actions:
        #     param_dict = {
        #         "reqIndex": param[0],
        #         "current_node_id": param[1],
        #         "next_node_id": param[2],
        #         "current_state": param[3],
        #         "done_episode": param[4],
        #         "timeSlot": param[5],
        #         "reward": param[6],
        #         "node_matrix": param[7].tolist(),
        #         "req_matrix": param[8].tolist(),
        #         "dist_matrix": param[9].tolist()
        #     }
        #     batch_json.append(param_dict)

        payload = {
            "successfulRequest": successful_requests,
            "timeSlot": timeSlot,
            "actions": [],
            "actionIds": actionIds
        }
        # print('**size of actions in update_reward ' , get_deep_size(actions) / (1024*1024) , ' MB with ' , len(actions) , ' actions')

        # print('size of payload in update_reward ' , get_deep_size(payload) / (1024*1024) , ' MB with ' , len(actions) , ' actions')

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                # print('update_reward api called successfully', timeSlot)
                # return response.json()
                print('~~~~~~~~~~@@@@@@update_reward api call completed', time.time() - t , 'sec for $$$$$$$$$$$$ timeSlot ' , timeSlot)

                return
        except httpx.RequestError as e:
            print(f"Network error while calling update_reward: {e}")
            return {"status": "error", "message": str(e)}
        except httpx.HTTPStatusError as e:
            print(f"HTTP error from update_reward API: {e.response.status_code}")
            return {"status": "error", "message": str(e)}
    def convert_to_serializable_actions(self , batch_params):
        """Convert any NumPy arrays or NumPy scalars to Python native types."""
        serializable_batch = []
        for action in batch_params:
            serializable_action = []
            for elem in action:
                if isinstance(elem, np.ndarray):
                    serializable_action.append(elem.tolist())
                elif isinstance(elem, (np.integer, np.floating)):
                    serializable_action.append(elem.item())
                else:
                    serializable_action.append(elem)
            serializable_batch.append(serializable_action)
        return serializable_batch
    def call_update_action_batch(self, batch_params: list):
        return
        url = "http://127.0.0.1:8000/update_action_batch"  # adjust host/port if needed

        """
        Send a batch of update_action calls in a single request.

        batch_params: list of dicts, each dict must contain keys:
            reqIndex, current_node_id, next_node_id, current_state, done_episode,
            timeSlot, reward, node_matrix, req_matrix, dist_matrix
        """
        # batch_params = self.convert_to_serializable_actions(batch_params)
        batch_json = []
        for param in batch_params:
            param_dict = {
                "reqIndex": param[0],
                "current_node_id": param[1],
                "next_node_id": param[2],
                "current_state": param[3],
                "done_episode": param[4],
                "timeSlot": param[5],
                "reward": param[6],
                "node_matrix": param[7].tolist(),
                "req_matrix": param[8].tolist(),
                "dist_matrix": param[9].tolist()
            }
            batch_json.append(param_dict)
            
        payload = {"batch": batch_json}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling batch API: {e}")
            return None
        
    def call_learn_and_predict_api(self , reqState, ent_matrix, req_matrix, dist_matrix, timeSlot):

        url = "http://127.0.0.1:8080/learn_predict"  # adjust host/port if needed
        reqId = str(uuid.uuid4())
        mpredis.set(f"reqId_{reqId}_ent_matrix", pickle.dumps(ent_matrix) , ex=300)  # expire in 5 minutes
        mpredis.set(f"reqId_{reqId}_req_matrix", pickle.dumps(req_matrix), ex=300)  # expire in 5 minutes
        mpredis.set(f"reqId_{reqId}_dist_matrix", pickle.dumps(dist_matrix), ex=300)  # expire in 5 minutes
        payload = {
            "reqIndex": reqState[4],
            "timeSlot": timeSlot,
            'reqId': reqId
        }
        # print('pppppppppaaaayyyyllllooooaaaaadddd in learn_predict ' , payload)
        # print('size of payload in learn_predict ' , get_deep_size(payload) / (1024*1024) , ' MB for reqIndex ' , reqState[4])
        # print('Calling learn_predict API with payload ===')
        try:
            t = time.time()
            response = requests.post(url, json=payload, timeout=5)
            # print('Time for learn_predict API call:', time.time() - t)
            t = time.time()
            response.raise_for_status()  # raises error for HTTP issues
            # print('Time after response.raise_for_status():', time.time() - t)
            t = time.time()
            result = response.json().get("result")
            # print('Time to parse JSON response:', time.time() - t)
            # print
            return result
        except Exception as e:
            print(f"Error calling learn_predict API: {e}")
            return None
        
    def get_action(self ,reqState , ent_matrix, req_matrix,dist_matrix, timeSlot):
        # return self.routingAgent.learn_and_predict_next_req_node_single(reqState , ent_matrix, req_matrix,dist_matrix)
        t = time.time()
        ret =  self.call_learn_and_predict_api(reqState , ent_matrix, req_matrix,dist_matrix, timeSlot)
        # print('============time to call learn_predict_api ' , time.time() - t)
        return ret


    def acquire_two_locks(self , lock1, lock2, timeout=2.0, backoff_range=(0.01, 0.1)):
        """
        Safely acquire two locks without deadlock.
        Always acquire in a fixed order.
        If not successful within timeout, release and retry.
        """
        start_time = time.time()
        while True:
            # Acquire first lock
            got_first = lock1.acquire(timeout=timeout)
            if not got_first:
                continue

            # Try to acquire the second lock
            got_second = lock2.acquire(timeout=timeout)
            if got_second:
                # success
                return True
            else:
                # Failed to acquire second, release first and back off
                lock1.release()
                sleep_time = random.uniform(*backoff_range)
                time.sleep(sleep_time)

            # Optional: break if too long
            if time.time() - start_time > timeout * 5:
                return False


    def route_schedule_single(self ,  args):
        # print('route_schedule_single called with algo#############################################:')
        node_matrix_info , req_matrix_info,dist_matrix , q_matrix, reqState, node_locks =  args

        shm_name, shape, dtype = node_matrix_info
        shm = shared_memory.SharedMemory(name=shm_name)
        ent_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        # self.shared_memories.append(shm)  # keep reference
        # print('=======matrix in route_schedule_single ' , ent_matrix.sum() )

        shm_name2, shape2, dtype2 = req_matrix_info
        shm2 = shared_memory.SharedMemory(name=shm_name2)
        req_matrix = np.ndarray(shape2, dtype=dtype2, buffer=shm2.buf)
        # self.shared_memories.append(shm2)  # keep reference
        # print('=======req_matrix in route_schedule_single ' , req_matrix.shape  )
        # agent = dill.loads(mpredis.get("routing_agent"))
        tt = time.time()
        # print('$$$$$$$$$$$$$$$$$$$time to load agent ' , time.time() - tt)
        """
        Serve only one request (reqState) using the routing agent.
        reqState: [src, dst, current_node, path, index, checked]
        """
        # topo = pickle.loads(serialized_topo)
        # nodes = pickle.loads(serialized_nodes)
        src, dst, current_node_id, path, index, checked = reqState
        next_node_id = None
        # current_node_id = current_node.id
        selectedNodes = [src]
        selectedEdges = []
        selectedlinks = []
        usedLinks = []
        prev_links = None
        hopCount = 0
        success = False
        good_to_search = True
        failed_no_ent = False
        failed_loop = False
        failed_swap = False
        fail_hopcount = False
        path = list(path)
        numtry = 0
        maxTry = self.maxTry
        fidelity = 1
        actions = []
        tl = time.time()
        swappSuccess = False
        action_time = 0
        a_id = 0

        while good_to_search and not success and numtry <= maxTry:
            # break
            # Get next action for this request
            # print('-------===----=-=-=-=-=going to get action ' , current_node_id , path , numtry)
            t = time.time()
            # with agent_lock:
            if True:
                # print('-------===----=-=-=-=-=acquired agent lock ' , current_node_id , path , numtry, reqState)
                # result = agent.learn_and_predict_next_req_node_single(reqState , ent_matrix, req_matrix,dist_matrix)
                # print('**going to get action for req ' , current_node_id , next_node_id)
                try:
                    result = self.get_action(reqState , ent_matrix, req_matrix,dist_matrix, self.timeSlot)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    exit(1)
                    # result = None
                # result = None
                action_time += time.time() - t
                if result is None:
                    # print('-------===----=-=-=-=-=no action found break' , current_node_id , path , numtry)
                    break
                # print('-------===----=-=-=-=-=got action ' , current_node_id ,next_node_id)
            # print('time to get action ' , time.time() - t)
            # result = agent.learn_and_predict_next_req_node_single(reqState)
            # if result is None:
            #         break
            # print('time to get action ' , time.time() - t)
            t = time.time()
            # with lock:
                # t1 = time.time()
            # shared_nodes = dill.loads(mpredis.get("shared_nodes"))
            # shared_nodes = self.topo.nodes
                # print('==shared_nodes load time ' , time.time() - t1)
                # el = 0
                # tk = 0
                # for n in shared_nodes:
                #     for l in n.links:
                #         if l.isEntangled(self.timeSlot):
                #             el += 1
                #         if l.notSwapped():
                #             tk += 1
                # print('==shared_nodes load time ' , time.time() - t1 , ' ent links ' , el, ' taken links ' , tk)
                # print('++++++process id:', os.getpid() , 'entering for processing')


            current_state, next_node_id = result
            current_state = (ent_matrix.copy().tolist(), req_matrix.copy().tolist())
            # next_node = dill.loads(mpredis.get("node_" + str(next_node_id)))

            # next_node = shared_nodes[next_node_id]
            self.tst.append(os.getpid())
            t1 = time.time()

                # shared_topo = update_shared_topo(os.getpid())
                # print('==shared_topo update time ' , time.time() - t1)
                # t1 = time.time()
                # shared_topo = dill.loads(mpredis.get("shared_topo"))
                # print('==shared_topo load time ' , time.time() - t1)
                # shared_topo.tst.append(os.getpid())

                # print('process id:', os.getpid(), 'thread id:', threading.get_ident() , set(shared_topo.tst))
            # current_node_id = current_node.id
            # current_node = shared_nodes[current_node.id]  # Get the current node object
            # current_node = dill.loads(mpredis.get("node_" + str(current_node.id)))
            if current_node_id == next_node_id:
                numtry += 1
                if numtry <= maxTry:
                    continue
                else:
                    good_to_search = False
                    failed_loop = True
                req_done = (not good_to_search)
                # print('added to path')

                reqState = (src,dst,next_node_id,tuple(path),index,req_done)
                self.requestState[index] = reqState
                req_matrix[index][2] = next_node_id
                req_matrix[index][5] = req_done
                req_matrix[len(self.requestState)+index][next_node_id] = 1
                continue
                
            cnlock = node_locks[current_node_id]
            nnlock = node_locks[next_node_id]
            # print('-------===----=-=-=-=-=going to acquire locks ' , current_node.id , next_node.id)
            # locks = [cnlock, nnlock]
            # if cnlock == nnlock:
            #     locks = [cnlock]
            # with lock in locks:
            pnode = current_node_id
            # for lock in {cnlock, nnlock}:
            #     lock.acquire()
            # with cnlock, nnlock:
            timeout = 1
            while True:
                if self.acquire_two_locks(cnlock, nnlock):

                    try:
                    
                        # print('-------===----=-=-=-=-=acquired locks+ ' ,  current_node_id , next_node_id , ' index: ' , index)
                        # Find entangled links
                        # ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                        # print(f"Processing request {src.id} to {dst.id},current node ID: {current_node.id} next node ID: {next_node_id}", 'len ent_links:', len(ent_links) , 'path:', path)
                        key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(current_node_id) + '_' + str(next_node_id)
                        # mpredis.set("shared_topo", dill.dumps(shared_topo))
                        # print('going to find ent_links for ' , (current_node_id , next_node_id) , ' ent_matrix ' , ent_matrix[current_node_id] [next_node_id])
                        # print('**going for ent_matrix ' , current_node_id , next_node_id , 'index' , index)
                        ent_links = ent_matrix[current_node_id] [next_node_id]
                        if not ent_links:

                            numtry += 1
                            if numtry <= maxTry:
                                continue
                            else:
                                good_to_search = False
                                failed_no_ent = True
                        else:
                            numtry = 0
                            ent_matrix[current_node_id] [next_node_id] -= 1
                            ent_matrix[next_node_id] [current_node_id] -= 1

                            # Fidelity check
                            # fidelity = self.fidelityAfterSwap(fidelity, ent_links[0].fidelity)
                            # if fidelity < self.topo.fidelity_threshold:
                            #     numtry += 1
                            #     if numtry <= maxTry:
                            #             continue
                            #     else:
                            #         numtry = 0
                            #         good_to_search = False


                        # Loop check
                        if next_node_id == current_node_id or next_node_id in path:
                            good_to_search = False
                            failed_loop = True
                            # break

                        # selectedNodes.append(next_node)
                        # selectedEdges.append((current_node, next_node))
                        # usedLinks.extend(prev_links)
                        path.append(next_node_id)
                        req_done = (not good_to_search) or success
                        # print('added to path')

                        reqState = (src,dst,next_node_id,tuple(path),index,req_done)
                        self.requestState[index] = reqState
                        req_matrix[index][2] = next_node_id
                        req_matrix[index][5] = req_done
                        req_matrix[len(self.requestState)+index][next_node_id] = 1
                        # print('Updated request state:', reqState)
                        # Success check
                        if next_node_id == dst.id and good_to_search:
                            success = True
                            good_to_search = False

                        # Prepare for next hop
                        # t1 = time.time()
                        # mpredis.set("node_" + str(current_node.id), dill.dumps(current_node))
                        # mpredis.set("node_" + str(next_node.id), dill.dumps(next_node))
                        # print('==shared_nodes save time ' , time.time() - t1)
                        pnode = current_node_id

                        current_node_id = next_node_id
                        # swappSuccess = True
                        # if success:
                        #     # print('going to swap for ' , (src.id , dst.id ))
                        #     for i in range(1 , len(path)-1):
                        #         swapped = False
                        #         # print('qmatrix ' , q_matrix)
                        #         if random.random() <= q_matrix[path[i]]:
                        #             swapped = True
                        #         if not swapped:
                        #             failed_swap = True
                        #             swappSuccess = False
                        #             # print('================failed swap==================')
                        #             break
                        


                        # if success and swappSuccess:
                        #     # print('going to find path for:', (src.id , dst.id ))
                        #     t2 = time.time()
                        #     for req in self.requests:
                        #             # src = req[0]
                        #             # dst = req[1]
                        #         if (src, dst) == (req[0], req[1]):
                        #                 # print('[REPS] finish time:', self.timeSlot - request[2])
                        #             self.requests.remove(req)
                        #             break

                        #     # successReq += 1
                        #     # totalEntanglement += 1


                        reward = -1

                        if req_done:
                            if success:
                                    # print("====success====" , src.id , dst.id , [n for n in path])
                                    # print('shortest path ----- ' , [n.id for n in targetPath])
                                reward = 10
                                    # reward = 1

                                total_fidelity += fidelity
                            else:
                                for i in range(1 , len(path)):
                                    ent_matrix[path[i-1]] [path[i]] += 1
                                    ent_matrix[path[i]] [path[i-1]] += 1
                            
                                # print("!!!!!!!=fail=!!!!!!!" , src.id , dst.id , [n for n in path] , 'threading.get_ident():', threading.get_ident())
                                    # print('shortest path ----- ' , [n.id for n in targetPath])
                                
                                # print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                                reward = -10

                    finally:
                        nnlock.release()
                        cnlock.release()
                        # print('-------===----=-=-=-=-=released locks* ' , pnode , next_node_id)
                    break
                else:
                    time.sleep(random.uniform(0.001, 0.005))


            T = [r for r in self.requestState if not r[5]]
            done_episode = (not good_to_search or success) and (len(T)==1)
            next_state = (ent_matrix.copy().tolist(), req_matrix.copy().tolist())
            actions.append([index , current_node_id , next_node_id , current_state  , done_episode ,self.timeSlot,reward, next_state, dist_matrix , a_id])
            a_id += 1
            # print('time for one hop2 ======== ' , time.time() - t , good_to_search , success , numtry , maxTry)
            
            # with lock2:
            #     self.routingAgent.update_action( reqState ,current_node_id,  next_node_id  , current_state  , done_episode)
        # print('time in action selection ======== ' , action_time,'s, for ' , len(actions), ' actions' )
        # print('=================final process id:', os.getpid() , 'time taken:', time.time()-tl)
       
        try:
            shm.close()  # Don't unlink, just close in worker
            shm2.close()
        except:
            pass
        return (success , actions , index , path)

    # def route_schedule_single(self ,  args):
        print('route_schedule_single called with algo#############################################:')
        node_matrix_info , reqState,lock , agent_lock ,reward_lock, node_locks =  args

        shm_name, shape, dtype = node_matrix_info
        shm = shared_memory.SharedMemory(name=shm_name)
        matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        print('=======matrix in route_schedule_single ' , matrix.sum() )

        # agent = dill.loads(mpredis.get("routing_agent"))
        tt = time.time()
        agent = self.routingAgent
        print('$$$$$$$$$$$$$$$$$$$time to load agent ' , time.time() - tt)
        """
        Serve only one request (reqState) using the routing agent.
        reqState: [src, dst, current_node, path, index, checked]
        """
        # topo = pickle.loads(serialized_topo)
        # nodes = pickle.loads(serialized_nodes)
        src, dst, current_node, path, index, checked = reqState
        selectedNodes = [src]
        selectedEdges = []
        selectedlinks = []
        usedLinks = []
        prev_links = None
        hopCount = 0
        success = False
        good_to_search = True
        failed_no_ent = False
        failed_loop = False
        failed_swap = False
        fail_hopcount = False
        path = list(path)
        numtry = 0
        maxTry = self.maxTry
        fidelity = 1
        actions = []
        tl = time.time()
        swappSuccess = False

        while good_to_search and not success and numtry <= maxTry:
            # break
            # Get next action for this request
            t = time.time()
            with agent_lock:
                result = agent.learn_and_predict_next_req_node_single(reqState)
                # result = None
                if result is None:
                    break
            # result = agent.learn_and_predict_next_req_node_single(reqState)
            # if result is None:
            #         break
            # print('time to get action ' , time.time() - t)
            t = time.time()
            # with lock:
                # t1 = time.time()
            # shared_nodes = dill.loads(mpredis.get("shared_nodes"))
            shared_nodes = self.topo.nodes
                # print('==shared_nodes load time ' , time.time() - t1)
                # el = 0
                # tk = 0
                # for n in shared_nodes:
                #     for l in n.links:
                #         if l.isEntangled(self.timeSlot):
                #             el += 1
                #         if l.notSwapped():
                #             tk += 1
                # print('==shared_nodes load time ' , time.time() - t1 , ' ent links ' , el, ' taken links ' , tk)
                # print('++++++process id:', os.getpid() , 'entering for processing')


            current_state, req_id, next_node_id, q, mask, valid_actions = result
            # next_node = dill.loads(mpredis.get("node_" + str(next_node_id)))

            next_node = shared_nodes[next_node_id]
            self.tst.append(os.getpid())
            t1 = time.time()

                # shared_topo = update_shared_topo(os.getpid())
                # print('==shared_topo update time ' , time.time() - t1)
                # t1 = time.time()
                # shared_topo = dill.loads(mpredis.get("shared_topo"))
                # print('==shared_topo load time ' , time.time() - t1)
                # shared_topo.tst.append(os.getpid())

                # print('process id:', os.getpid(), 'thread id:', threading.get_ident() , set(shared_topo.tst))
            current_node_id = current_node.id
            current_node = shared_nodes[current_node.id]  # Get the current node object
            # current_node = dill.loads(mpredis.get("node_" + str(current_node.id)))
            cnlock = node_locks[current_node.id]
            nnlock = node_locks[next_node.id]
            # print('-------===----=-=-=-=-=going to acquire locks ' , current_node.id , next_node.id)
            # locks = [cnlock, nnlock]
            # if cnlock == nnlock:
            #     locks = [cnlock]
            # with lock in locks:
            for lock in {cnlock, nnlock}:
                lock.acquire()
            try:
                # print('-------===----=-=-=-=-=acquired locks ' , current_node.id , next_node.id)
                # Find entangled links
                ent_links = [link for link in current_node.links if (link.isEntangled(self.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                # print(f"Processing request {src.id} to {dst.id},current node ID: {current_node.id} next node ID: {next_node_id}", 'len ent_links:', len(ent_links) , 'path:', path)
                key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                # mpredis.set("shared_topo", dill.dumps(shared_topo))
                
                if not ent_links:

                    numtry += 1
                    if numtry <= maxTry:
                        continue
                    else:
                        good_to_search = False
                        failed_no_ent = True
                else:
                    ent_links = [ent_links[0]]
                    ent_links[0].taken = True
                    selectedlinks.append(ent_links[0])
                    prev_links = [ent_links[0]]
                    numtry = 0

                    # Fidelity check
                    fidelity = self.fidelityAfterSwap(fidelity, ent_links[0].fidelity)
                    if fidelity < self.topo.fidelity_threshold:
                        numtry += 1
                        if numtry <= maxTry:
                                continue
                        else:
                            numtry = 0
                            good_to_search = False


                # Loop check
                if next_node == current_node or next_node.id in path:
                    good_to_search = False
                    failed_loop = True
                    # break

                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))
                # usedLinks.extend(prev_links)
                path.append(next_node.id)
                req_done = (not good_to_search) or success

                reqState = (src,dst,next_node,tuple(path),index,req_done)
                self.requestState[index] = reqState
                # print('Updated request state:', reqState)
                # Success check
                if next_node.id == dst.id and good_to_search:
                    success = True
                    good_to_search = False

                # Prepare for next hop
                # t1 = time.time()
                # mpredis.set("node_" + str(current_node.id), dill.dumps(current_node))
                # mpredis.set("node_" + str(next_node.id), dill.dumps(next_node))
                # print('==shared_nodes save time ' , time.time() - t1)
                current_node = next_node
                swappSuccess = True
                if success:
                    print('going to swap for ' , (src.id , dst.id ))
                    for i in range(len(selectedlinks)-1):
                        l1 = selectedlinks[i]
                        l2 = selectedlinks[i+1]
                        n1 = l1.n1
                        n2 = l1.n2
                        n = n1 if l2.contains(n1) else n2
                        swapped = n.attemptSwapping(l1,l2)

                        usedLinks.append(l1)
                        usedLinks.append(l2)
                        if not swapped:
                            failed_swap = True
                            swappSuccess = False
                            print('================failed swap==================')
                            break
                


                if success and swappSuccess:
                    print('going to find path for:', (src.id , dst.id ))
                    t2 = time.time()
                    for req in self.requests:
                            # src = req[0]
                            # dst = req[1]
                        if (src, dst) == (req[0], req[1]):
                                # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(req)
                            break

                    # successReq += 1
                    # totalEntanglement += 1


                reward = -1

                if req_done:
                    if success:
                            # print("====success====" , src.id , dst.id , [n for n in path])
                            # print('shortest path ----- ' , [n.id for n in targetPath])
                        reward = 10
                            # reward = 1

                        total_fidelity += fidelity
                    else:
                        for link in selectedlinks:
                            link.taken = False
                        print("!!!!!!!=fail=!!!!!!!" , src.id , dst.id , [n for n in path] , 'threading.get_ident():', threading.get_ident())
                            # print('shortest path ----- ' , [n.id for n in targetPath])
                        
                        print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                        reward = -10


                    # print('lenT ' , len(T))

                # key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(prev_node.id) + '_' + str(next_node.id)

                    # reward = -self.topo.numOfRequestPerRound
                
                for link in usedLinks:
                    link.clearPhase4Swap()
                # print('===============process id:', os.getpid() , 'leaving after processing, time taken:', time.time()-tl)  
                # t1 = time.time()
                # mpredis.set("shared_nodes", dill.dumps(shared_nodes))
                # print('==shared_nodes save time ' , time.time() - t1)
            finally:
                for lock in {cnlock, nnlock}:
                    lock.release()
            
            with reward_lock:
                t1 = time.time()
                # reward_routing = dill.loads(mpredis.get("reward_routing"))
                reward_routing = self.topo.reward_routing
                # print('==reward_routing load time ' , time.time() - t1)
                try:
                    reward_routing[key] += reward
                except:
                    reward_routing[key] = reward
                t1 = time.time()
                # mpredis.set("reward_routing", dill.dumps(reward_routing))  
                # print('==reward_routing save time ' , time.time() - t1)  

                
            T = [r for r in self.requestState if not r[5]]
            done_episode = (not good_to_search or success) and (len(T)==1)
            actions.append((reqState , current_node_id , next_node_id , current_state  , done_episode))
            
            # with lock2:
            #     self.routingAgent.update_action( reqState ,current_node_id,  next_node_id  , current_state  , done_episode)
            
        print('=================final process id:', os.getpid() , 'time taken:', time.time()-tl)
        return (success and swappSuccess , actions)

        

    


    def findPathForDQRL(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        if self.DijkstraForDQRL(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]
            path = path[::-1]
            return path
        else:
            return []
    
    def DijkstraForDQRL(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if 0 < self.edgeSuccessfulEntangle(node1, node2):
                    adjcentList[node1].add(node2)
        
        # for node in adjcentList:
        #     print([n.id for n in adjcentList[node]])
        
        distance = {node : math.inf for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        # print(src.id)
        pq.put((self.weightOfNode[src], src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = distance[u] + self.weightofLink(u,next)
                if distance[next] > newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((distance[next], next.id))

        return False
    def weightofLink(self , u , v):
        # w1 = .5
        # w2 = .5

        w1 = self.w1
        w2 = self.w2
        if u == v:
            return 0
        capacity = 0
        fidelity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken:
                capacity += 1
                fidelity = max(fidelity , link.fidelity)

        return w1/capacity + w2 * (-math.log(fidelity))
        # return w1/capacity + w2/fidelity
    def edgeCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        used = 0
        for SDpair in self.srcDstPairs:
            used += self.fi[SDpair][(u, v)]
            used += self.fi[SDpair][(v, u)]
        return capacity - used

    def widthForSort(self, path):
        # path[-1] is the path of weight
        return -path[-1]
    
    def PFT(self):

        # initialize fi and ti
        self.fi = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti = {SDpair : 0 for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for u in self.topo.nodes:
                for v in self.topo.nodes:
                    self.fi[SDpair][(u, v)] = 0
        
        # PFT
        failedFindPath = False
        while not failedFindPath:
            self.LP1()
            failedFindPath = True
            Pi = {}
            paths = []
            for SDpair in self.srcDstPairs:
                Pi[SDpair] = self.findPathsForPFT(SDpair)

            for SDpair in self.srcDstPairs:
                K = len(Pi[SDpair])
                for k in range(K):
                    width = math.floor(Pi[SDpair][k][-1])
                    Pi[SDpair][k][-1] -= width
                    paths.append(Pi[SDpair][k])
                    pathLen = len(Pi[SDpair][k]) - 1
                    self.ti[SDpair] += width
                    if width == 0:
                        continue
                    failedFindPath = False
                    for nodeIndex in range(pathLen - 1):
                        node = Pi[SDpair][k][nodeIndex]
                        next = Pi[SDpair][k][nodeIndex + 1]
                        self.fi[SDpair][(node, next)] += width

            sorted(paths, key = self.widthForSort)

            for path in paths:
                pathLen = len(path) - 1
                width = path[-1]
                SDpair = (path[0], path[-2])
                isable = True
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    if self.edgeCapacity(node, next) < 1:
                        isable = False
                
                if not isable:
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                    continue
                
                failedFindPath = False
                self.ti[SDpair] += 1
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    self.fi[SDpair][(node, next)] += 1

        # print('[REPS] PFT end')
        for SDpair in self.srcDstPairs:
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                need = self.fi[SDpair][(u, v)] + self.fi[SDpair][(v, u)]
                if need:
                    assignCount = 0
                    for link in u.links:
                        if link.contains(v) and link.assignable():
                            # link(u, v) for u, v in edgeIndices)
                            link.assignQubits()
                            self.totalUsedQubits += 2
                            assignCount += 1
                            if assignCount == need:
                                break 
          
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken:
                capacity += 1
        # print(capacity)
        return capacity
    def edgeSuccessfulEntangleForELS(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled and not link.taken and not link.considered:
                capacity += 1
        # print(capacity)
        return capacity

    

    

    def ELS(self , Pi):
        Ci = self.pathForELS
        self.y = {(u, v) : 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        needLink = {}
        nextLink = {node : [] for node in self.topo.nodes}
        T = [SDpair for SDpair in self.srcDstPairs]
        for sd in Pi:
            if len(Pi[sd]) and sd in T:
                T.remove(sd)
        output = []
        while len(T) > 0 :
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangleForELS(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths:
                    Ci[SDpair].remove(path)
                if len(Ci[SDpair]) == 0 and SDpair in T:
                    T.remove(SDpair)
            
            if len(T) == 0:
                break

            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) < minLength:
                        minLength = len(path)
                        i = SDpair
            
            src = i[0]
            dst = i[1]

            minR = math.inf
            for path in Ci[i]:
                r = 0
                for node in path:
                    r += self.weightOfNode[node]
                if minR > r:
                    targetPath = path
                    minR = r
            
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []

            Pi[i].append(targetPath)
            output.append(targetPath)
            if len(targetPath) ==2:
                for link in targetPath[0].links:
                    if link.contains(targetPath[1]) and link.entangled and link.notSwapped() and not link.taken and not link.considered:
                        targetLink1 = link
                        break
                targetLink1.considered = True
            for nodeIndex in range(1, len(targetPath) - 1):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                for link in node.links:
                    if link.contains(next) and link.entangled and link.notSwapped() and not link.taken and not link.considered:
                        targetLink1 = link
                    
                    if link.contains(prev) and link.entangled and link.notSwapped() and not link.taken and not link.considered:
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1

                nextLink[node].append(targetLink1)
                targetLink1.considered = True
                if nodeIndex == 1:
                    targetLink2.considered = True
                needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))

            T.remove(i)
        print('** before graph ' , len(output))
        print([(path[0].id , path[-1].id) for path in output])

        print('** after graph ' , len(output))





        return len(output)

    def filterReqeuest(self):
        self.requests = list(filter(lambda x: self.timeSlot -  x[2] < self.topo.requestTimeout -1 , self.requests))

    def findPathsForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForPFT(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForPFT(path, SDpair)
            
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fi_LP[SDpair][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node in self.topo.nodes:
            for link in node.links:
                neighbor = link.theOtherEndOf(node)
                adjcentList[node].add(neighbor)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fi_LP[SDpair][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], next.id))

        return False

    def widthForPFT(self, path, SDpair):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fi_LP[SDpair][(currentNode, nextNode)])

        return width
    
    def findPathsForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForEPS(SDpair, k):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForEPS(path, SDpair, k)
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fki_LP[SDpair][k][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.edgeSuccessfulEntangle(node1, node2) > 0:
                    adjcentList[node1].add(node2)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fki_LP[SDpair][k][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], next.id))

        return False

    def widthForEPS(self, path, SDpair, k):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fki_LP[SDpair][k][(currentNode, nextNode)])

        return width
    
    def findPathForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        if self.DijkstraForELS(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]
            path = path[::-1]
            return path
        else:
            return []
    
    def DijkstraForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.y[(node1, node2)] < self.edgeSuccessfulEntangle(node1, node2):
                    adjcentList[node1].add(node2)
        
        distance = {node : math.inf for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((self.weightOfNode[src], src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = dist
            visited[u] = True
            
            for next in adjcentList[u]:
                # newDistance = distance[u] + self.weightOfNode[next]
                newDistance = distance[u] + self.weightofLink(u,next)
                if distance[next] > newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((distance[next], next.id))

        return False
if __name__ == '__main__':
    
    topo = Topo.generate(50, 0.9, 5, 0.0002, 6)
    s = DQRL(topo)
    result = AlgorithmResult()
    samplesPerTime = 8 * 2
    ttime = 100
    rtime = ttime
    # requests = {i : [] for i in range(ttime)}

    # for i in range(ttime):
    #     if i < rtime:

    #         # ids =  [(1,15), (1,16), (4,17), (3,16)]

    #         # for (p,q) in ids:
    #         #     source = None
    #         #     dest = None
    #         #     for node in topo.nodes:

    #         #         if node.id == p:
    #         #             source = node
    #         #         if node.id == q:
    #         #             dest = node
    #         #     requests[i].append((source , dest))

    #         a = sample(topo.nodes, samplesPerTime)
    #         for n in range(0,samplesPerTime,2):
    #             requests[i].append((a[n], a[n+1]))
    #     print('[REPS] S/D:' , i , [(a[0].id , a[1].id) for a in requests[i]])

    # for i in range(ttime):
    #     result = s.work(requests[i], i)
    

    for i in range(0, 100):
        requests = []
        if i < 100:
            for j in range(20):
                a = sample(topo.nodes, 2)
                requests.append((a[0], a[1]))
            
            # ids = [(1,15), (1,16), (4,17), (3,16)]
            # for (p,q) in ids:
            #     source = None
            #     dest = None
            #     for node in topo.nodes:

            #         if node.id == p:
            #             source = node
            #         if node.id == q:
            #             dest = node
            #     requests.append((source , dest))

            s.work(requests, i)
        else:
            s.work([], i)

    # print(result.waitingTime, result.numOfTimeslot)