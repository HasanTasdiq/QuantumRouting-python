
from .Node import Node
import random
import math

class Link:
    
    def __init__(self, topo, n1: Node, n2: Node, s1: bool, s2: bool, id: int, l: float):
        self.id, self.n1, self.n2, self.s1, self.s2, self.alpha = id, n1, n2, s1, s2, topo.alpha
        self.assigned = False
        self.entangled = False
        self.p = math.exp(-self.alpha * l)
        # print(self.n1.id, self.n2.id, self.p)

    def theOtherEndOf(self, n: Node): 
        if (self.n1 == n): 
            tmp = self.n2
        elif (self.n2 == n): 
            tmp = self.n1 
        return tmp
    def contains(self, n: Node):  
        return self.n1 == n or self.n2 == n
    def swappedAt(self, n: Node): 
        return (self.n1 == n and self.s1 or self.n2 == n and self.s2)
    def swappedAtTheOtherEndOf(self, n: Node):  
        return (self.n1 == n and self.s2 or self.n2 == n and self.s1)
    def swapped(self):  
        return self.s1 or self.s2
    def notSwapped(self):  
        return not self.swapped()


    def assignQubits(self):
        # prevState = self.assigned
        self.assigned = True
        self.n1.remainingQubits -= 1
        self.n2.remainingQubits -= 1
  
    def clearEntanglement(self):
        preState = self.assigned
        self.assigned = False
        self.entangled = False

        if preState:
            self.n1.remainingQubits += 1
            self.n2.remainingQubits += 1
   
    
    def tryEntanglement(self):
        b = self.assigned and self.p >= random.random()
        self.entangled = b
        return b
  
    def assignable(self): 
        return not self.assigned and self.n1.remainingQubits > 0 and self.n2.remainingQubits > 0
