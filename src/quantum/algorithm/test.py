import os
import gurobipy as gp

import networkx as nx
import numpy as np
import random
from itertools import combinations

# algoNames = ['SEER1' , 'SEER2']

# print('SEER' in algoNames[0])
td = {}
td[1] = 0
path = [1 , 2, 3, 4, 5]
l2 = path[1:]
dp = [[0]*5]*5


# print([(p , q) for (p , q) in  zip(path[0:-1] , path[1:])])

max_ = 0
for i in range(10000000):
    rand = int(random.random()*5+3) 
#     print(rand)
    max_ = max(max_ , rand)
print(max_)

