import itertools
import sys
sys.path.append("../")
import networkx as nx
from topo.Topo import Topo
from random import sample
import copy
from itertools import islice
import time


def get_k_paths(G , source ,target):
    print('in get k path' , source , target)
    retPath = []
    k = 2
    try:
        print('trying for path')
        # paths = list(nx.shortest_simple_paths(G, source, target))
        paths =  list(
            islice(nx.shortest_simple_paths(G, source, target), k)
        )
    except:
        print('Exception')
        return []
    # print('in g p ::: ', [p for p in paths])
    i = 0
    for path in paths:
        print(path)
        retPath.append(path)
        i += 1
        if i >= k:
            break
    # print('in g p ret  ::: ', [p for p in retPath])
    
    return retPath

gall_paths = []

def get_total_path(reqs , all_paths ):
    # print('get_total_path' , reqs)
    # for comb in all_paths:
    #     print([path for path in comb[1:]])

    global gall_paths
    if not len(reqs):
        return
    req = reqs[0]
    reqs.pop(0)
    new_all_paths = all_paths
    new_path_found = True
    for comb in all_paths:
        G = comb[0]
        last_path = comb[-1]



        paths = get_k_paths(G , req[0] , req[1])
        if not len(paths):
            new_path_found = False
            continue
        # print(req)
        # print(paths)
        
        new_all_paths.remove(comb)

        for path in paths:
            G2 = copy.deepcopy(G)
            # print(G2.edges())
            # print(path)
            for i in range(len(path) - 1):
                edge = (min((path[i] , path[i+1])) , max((path[i] , path[i+1])))
                G2.remove_edge(edge[0]  , edge[1])
            comb2 = copy.deepcopy(comb)
            comb2[0] = G2
            # comb2.extend([path])
            comb2.append(path)
            # print('comb2 ' , comb2[1:])
            new_all_paths.append(comb2)
    # if not new_path_found:
    #     gall_paths = all_paths
    #     # gall_paths.extend(new_all_paths)

    #     return get_total_path(reqs , gall_paths)
        
    # all_paths = new_all_paths
    gall_paths = new_all_paths

    return get_total_path(reqs , gall_paths)

def shortest_approach(G , reqs):
    paths = []
    for r in reqs:
        try:
            path = nx.shortest_path(G , r[0] , r[1])
        except:
            path = []
        if len(path):
            paths.append(path)
        for i in range(len(path) - 1):
            edge = (min((path[i] , path[i+1])) , max((path[i] , path[i+1])))
            G.remove_edge(edge[0]  , edge[1])
    return paths


nodeNo = 20
degree = 6
times = 1
numOfRequestPerRound = 10
topo = Topo.generate(nodeNo, 0.9, 5,0.002, degree)
G = topo.updatedG()
print(G.nodes())
print(G.edges())
ids = {i : [] for i in range(times)}


for t in range(times):
    for _ in range(numOfRequestPerRound):

        while True:
            a = sample([i for i in range(nodeNo)], 2)
            if (not ((a[0], a[1]) in ids)) and (not ((a[1], a[0]) in ids)):
                break
        ids[t].append((a[0], a[1]))

for t in range(times):
    reqs = copy.deepcopy(ids[t])
    print(reqs)

    req = reqs[0]
    reqs.pop(0)
    all_paths = []
    paths = get_k_paths(G , req[0] , req[1])
    print('in main')
    print([p for p in paths])
    for path in paths:
        # print('in main,  path:' , path)
        G2 = copy.deepcopy(G)
        # print(G2.nodes())
        # print(G2.edges())
        for i in range(len(path) - 1):
            edge = (min((path[i] , path[i+1])) , max((path[i] , path[i+1])))
            G2.remove_edge(edge[0]  , edge[1])
        comb = [G2 , path]
        all_paths.append(comb)

    for comb in all_paths:
        print([path for path in comb[1:]])

    print('before recursive\n')
    get_total_path(reqs , all_paths)

    print('--------------')
    lc = []
    pl = []
    opt_comb = []
    
    for comb in gall_paths:
        # print('-------path in comb-------')

        # print([path for path in comb[1:]])
        lc.append(len(comb)-1)
        if len(opt_comb) < len(comb)-1:
            opt_comb = comb[1:]
        l = 0
        for path in comb[1:]:
            l+= len(path)
        pl.append(l)
    print(min(lc) , max(lc))
    print(min(pl) , max(pl))
    # print(gall_paths[0][1:])
    print(ids[t])
    print([path for path in opt_comb])




    s_path = shortest_approach(G , ids[t])

    print('-------shortest -------')
    print(len(s_path))
    l = 0
    for path in s_path:
            l+= len(path)
    print(l)










    
    
        



