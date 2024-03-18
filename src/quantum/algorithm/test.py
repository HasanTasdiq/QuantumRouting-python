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




def generateParenthesis(n):
        """
        :type n: int
        :rtype: List[str]
        """
        def paren(left , right , current):
                if len(current) == 2 *n:
                        res.append(current)
                        return
                if left < n:
                        paren(left +1 , right , current + '(')
                if right < left:
                        paren(left , right +1 , current + ')')
                        

        
        res = []
        paren(0 , 0 , '')
        return res

print(generateParenthesis(4))
