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

def get_n_items(sorted_list, n):
    if len(sorted_list) < 2 or n < 2:
        return sorted_list[:n]

    common_diff = sorted_list[1] - sorted_list[0]
    step_size = (sorted_list[-1] - sorted_list[0]) / (n - 1)
    
    result = []
    i = 0
    while len(result) < n:
        if i >= len(sorted_list):
            i = len(sorted_list) - 1
        print(i)
        result.append(sorted_list[i])
        i += int(round(step_size / common_diff))
    return result
sorted_list = [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0, 11500.0, 12000.0, 12500.0, 13000.0, 13500.0, 14000.0, 14500.0, 15000.0, 15500.0, 16000.0, 16500.0, 17000.0, 17500.0, 18000.0, 18500.0, 19000.0, 19500.0, 20000.0, 20500.0, 21000.0, 21500.0, 22000.0, 22500.0, 23000.0, 23500.0, 24000.0, 24500.0, 25000.0, 25500.0, 26000.0, 26500.0, 27000.0, 27500.0, 28000.0, 28500.0, 29000.0, 29500.0]
print(len(sorted_list))
n = 5
print(get_n_items(sorted_list, n))
# print(np.random.randint(0, 2))


