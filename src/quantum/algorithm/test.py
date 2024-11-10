import itertools
import sys
sys.path.append("../")
import networkx as nx
from topo.Topo import Topo
from random import sample
import copy
from itertools import islice
import time

nodeNo = 30
degree = 1
times = 100
numOfRequestPerRound = 40
topo = Topo.generate(nodeNo, 0.9, 5,0.002, degree)
# G = topo.updatedG()
# print(G.nodes())
# print(G.edges())
ids = {i : [] for i in range(times)}
total_succ = 0
total_length = 0

total_succ_shortest = 0
total_length_shortest = 0


def get_k_paths_(G , source ,target):
    # print('in get k path' , source , target)
    retPath = []
    k = 2
    try:
        # print('trying for path')
        # paths = list(nx.shortest_simple_paths(G, source, target))
        # paths =  list(
        #     islice(nx.shortest_simple_paths(nx.Graph(G), source, target), k)
        # )
        paths =  list(
            islice(nx.all_simple_paths(G, source, target), k)
        )
    except:
        # print('Exception')
        return []
    # print('in g p ::: ', [p for p in paths])
    i = 0
    for path in paths:
        # print(path)
        retPath.append(path)
        i += 1
        if i >= k:
            break
    # print('in g p ret  ::: ', [p for p in retPath])
    
    return retPath

def get_k_paths(G, source , target):
    G2 = copy.deepcopy(G)
    ret = []
    k = 2
    for _ in range(k):
        try:
            path = list(nx.shortest_path(G2 , source, target))
        except:
            path = []
        if len(path):
            ret.append(path)
        for i in range(len(path) - 1):
            edge = (min((path[i] , path[i+1])) , max((path[i] , path[i+1])))
            # print('before remove ' , len(G2.edges()))
            G2.remove_edge(edge[0]  , edge[1])
            # print('after remove ' , len(G2.edges()))

    return ret

gall_paths = []

def get_total_path(reqs , all_paths ):
    print('get_total_path' , reqs)
    # for comb in all_paths:
    #     print([path for path in comb[1:]])

    global gall_paths
    if not len(reqs):
        return
    req = reqs[0]
    reqs.pop(0)
    new_all_paths = []
    new_path_found = True
    for comb in all_paths:
        G = comb[0]
        last_path = comb[-1]



        paths = get_k_paths(G , req[0] , req[1])
        if not len(paths):
            new_path_found = False
            new_all_paths.append(copy.deepcopy(comb))
            continue
        # print(req)
        # print(paths)
        # print(len(all_paths))
        
        # new_all_paths.remove(comb)
        # print('after remove ',len(all_paths))


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





for t in range(times):
    for _ in range(numOfRequestPerRound):

        while True:
            a = sample([i for i in range(nodeNo)], 2)
            if (not ((a[0], a[1]) in ids)) and (not ((a[1], a[0]) in ids)):
                break
        ids[t].append((a[0], a[1]))

for t in range(times):
    reqs = copy.deepcopy(ids[t])
    # print(reqs)
    G = topo.updatedG_all()
    req = reqs[0]
    reqs.pop(0)
    all_paths = []
    paths = get_k_paths(G , req[0] , req[1])
    # print('in main')
    # print([p for p in paths])
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

    # for comb in all_paths:
    #     print([path for path in comb[1:]])

    # print('before recursive\n')
    get_total_path(reqs , all_paths)

    print('--------------' , t , '----------')
    lc = []
    pl = []
    opt_comb = []
    
    for comb in gall_paths:
        # print('-------path in comb-------')

        # print([path for path in comb[1:]])
        lc.append(len(comb)-1)
        if len(opt_comb) < len(comb)-1:
            opt_comb = comb[1:]
        # l = 0
        # for path in comb[1:]:
        #     l+= len(path)
        # pl.append(l)
    for comb in gall_paths:
        if len(comb) -1 == max(lc):
            l = 0
            for path in comb[1:]:
                l+= len(path)
            pl.append(l)
    print(max(lc))
    print(min(pl) , max(pl))
    # print(gall_paths[0][1:])
    # print(ids[t])
    # print([path for path in opt_comb])

    total_succ += max(lc)
    total_length += min(pl)




    s_path = shortest_approach(G , ids[t])

    print('-------shortest -------')
    print(len(s_path))
    l = 0
    for path in s_path:
            l+= len(path)
    print(l)

    total_succ_shortest += len(s_path)
    total_length_shortest += l


print('=======FInal result =======')
print(total_succ/ times , total_length/total_succ , '\n')
print(total_succ_shortest/ times , total_length_shortest/total_succ_shortest)










    
    
        


