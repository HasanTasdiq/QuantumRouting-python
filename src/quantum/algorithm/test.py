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





def longestPalindrome( s):
        """
        :type s: str
        :rtype: str
        """
        max_ = 0
        max_i = 0
        max_j = 0
        dp = []
        for i in range(len(s)):
             r = []
             for j in range(len(s)):
                  r.append(0)
             dp.append(r)
        
        for i in range(len(s)):
            for j in range(len(s)):
                if s[i] == s[-j -1]:
                    if i>0 and j>0:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = 1
                else:
                    dp[i][j] = 0
                if dp[i][j] > max_:
                    max_ = dp[i][j]
                    max_i = i
                    max_j = j
        res = s[max_i - max_ + 1 : max_i+1]

        return res

print(longestPalindrome('babad'))