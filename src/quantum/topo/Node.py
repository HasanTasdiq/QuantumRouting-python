

import random

class Node:

    
    def __init__(self, id: int, loc: tuple, nQubits: int, topo) -> bool:
        self.id = id
        self.loc = loc
        self.remainingQubits = nQubits
        self.q = topo.q
        self.internalLinks = []

    def attemptSwapping(self, l1, l2):  # l1 -> Link, l2 -> Link
        if l1.n1 == self:    
            l1.s1 = True
        else:       
            l1.s2 = True
        
        if l2.n1 == self:    
            l2.s1 = True
        else: 
            l2.s2 = True
        
        b = random.random() <= self.q
        if b:
            self.internalLinks.append((l1, l2))
        return b

        