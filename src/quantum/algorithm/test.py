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

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def swapPairs( head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = head

        current = prev.next
        print(current.val)
        # prevprev = None
        # while current is not None:
        #     prev.next = current.next
        #     current.next = prev
        #     prevprev = prev

        #     prev = current.next

head = ListNode(val= 1, next= ListNode(val= 2, next= ListNode(val= 3, next= ListNode(val= 4, next= None))))
# current = head
# for i in range(1 , 4):
#      n = ListNode(i)
#      current.next = n
#      current = n

     
swapPairs(head)

