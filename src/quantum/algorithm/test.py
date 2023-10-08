import os
import gurobipy as gp
from gurobipy import quicksum
import networkx as nx
import numpy as np

# algoNames = ['SEER1' , 'SEER2']

# print('SEER' in algoNames[0])


bias_weights = [x%10==0 for x in range(100)]
prob = np.array(bias_weights) / np.sum(bias_weights)
print(prob)

for _ in range(20):
    sample_size = 2
    choice_indices = np.random.choice(len(prob), size=sample_size, replace=False, p=prob)

# Get corresponding paths
    print(choice_indices)