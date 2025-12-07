from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.managers import BaseManager
import os
import threading
import time
from .Node import sNode
from .Link import sLink
from multiprocessing import Lock
import logging, multiprocessing, os
import redis, dill
from multiprocessing import Lock, Manager
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import random

# mpredis = redis.Redis()
mpredis = redis.Redis(host='localhost', port=6379, db=0)
lock1 = None
agent_lock = None
reward_lock = None
node_locks = {}

# qManager = Manager()
# for i in range(100):
#     node_locks[i] = qManager.Lock()


logging.basicConfig(
    level=logging.INFO,
    format="%(processName)s [PID=%(process)d] %(message)s"
)

class qManager(BaseManager): pass
qManager.register('sNode', sNode,)
qManager.register('sLink', sLink)

executor = ProcessPoolExecutor(max_workers=50)
# executor = ThreadPoolExecutor(max_workers=8)

train_executor = ThreadPoolExecutor(max_workers=20)


def route_schedule_single2(  reqState):
    print('route_schedule_single called with algo#############################################:')


def getLock1():
    global lock1

    if lock1 is None:
        lock1 = Manager().Lock()
    return lock1

def update_shared_topo(task_id):
    print('update_shared_topo called with task_id:', task_id)
    with mpredis.pipeline() as pipe:
        while True:
            try:
                # Watch the key for changes
                pipe.watch("shared_topo")

                # Load the object
                obj = dill.loads(pipe.get("shared_topo"))

                # Modify it
                obj.tst.append(task_id)

                # Start transaction
                pipe.multi()
                pipe.set("shared_topo", dill.dumps(obj))
                pipe.execute()  # commits if key unchanged
                return obj
                break
            except redis.WatchError:
                # Retry if another process modified it concurrently
                continue
def solve_max_throughput_with_paths(G, pairs, time_limit=120):
    G = nx.MultiDiGraph(G)
    """
    Solves Maximum Edge-Disjoint Paths on any MultiDiGraph.
    """
    model = gp.Model("RandomGraphRouting")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    
    # --- 1. Variables ---
    served = {}
    x = {}
    edge_list = list(G.edges(keys=True))
    
    for k in range(len(pairs)):
        served[k] = model.addVar(vtype=GRB.BINARY, name=f"served_{k}")
        for u, v, key in edge_list:
            x[k, u, v, key] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{u}_{v}_{key}")
            
    # --- 2. Constraints ---
    
    # A. Flow Conservation
    for k, (s, t) in enumerate(pairs):
        for n in G.nodes():
            flow_out = gp.quicksum(x[k, n, v, key] for _, v, key in G.out_edges(n, keys=True))
            flow_in  = gp.quicksum(x[k, u, n, key] for u, _, key in G.in_edges(n, keys=True))
            
            if n == s:
                model.addConstr(flow_out - flow_in == served[k])
            elif n == t:
                model.addConstr(flow_out - flow_in == -served[k])
            else:
                model.addConstr(flow_out - flow_in == 0)

    # B. Edge Capacity (Disjointness)
    for u, v, key in edge_list:
        model.addConstr(gp.quicksum(x[k, u, v, key] for k in range(len(pairs))) <= 1)

    # --- 3. Objective ---
    total_served = gp.quicksum(served[k] for k in range(len(pairs)))
    total_hops   = gp.quicksum(x[k, u, v, key] for k in range(len(pairs)) for u, v, key in edge_list)
    
    model.setObjective(1.0 * total_served - 0.001 * total_hops, GRB.MAXIMIZE)

    # --- 4. Solve ---
    # print(f"Optimizing topology with {len(G.edges)} links...")
    model.optimize()

    # --- 5. Extract Paths ---
    final_paths = {}
    if model.status != GRB.INFEASIBLE:
        for k in range(len(pairs)):
            if served[k].X > 0.5:
                # Trace path
                active_edges = []
                for u, v, key in edge_list:
                    if x[k, u, v, key].X > 0.5:
                        active_edges.append((u, v))
                
                s, t = pairs[k]
                if not active_edges: continue
                
                path = [s]
                curr = s
                next_map = {u: v for u, v in active_edges}
                
                # Protect against infinite loops if solver returns cycles (unlikely with hop penalty)
                steps = 0
                while curr != t and steps < len(G.nodes):
                    if curr in next_map:
                        curr = next_map[curr]
                        path.append(curr)
                        steps += 1
                    else:
                        break
                final_paths[k] = path

    return final_paths

def solve_max_throughput_ILP(G, pairs):
    routes = solve_max_throughput_with_paths(G, pairs, time_limit=180)

    print(f"\n--- Results ---")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Total Links (inc. parallel): {len(G.edges)}")
    print(f"Requests: {len(pairs)}")
    print(f"Served: {len(routes)}")
    print(f"Success Rate: {len(routes)/len(pairs)*100:.1f}%")