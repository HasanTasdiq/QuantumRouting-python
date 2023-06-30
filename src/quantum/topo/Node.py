

import random

class Node:

    
    def __init__(self, id: int, loc: tuple, nQubits: int, topo) -> bool:
        self.id = id
        self.loc = loc
        self.remainingQubits = int(nQubits)
        self.q = topo.q
        self.internalLinks = []
        self.prevInternalLinks = []
        self.neighbors = [] 
        self.links = [] 

    def attemptSwapping(self, l1, l2 , times = 1):  # l1 -> Link, l2 -> Link

    

        if l1.n1 == self:    
            l1.s1 = True
        else:       
            l1.s2 = True
        
        if l2.n1 == self:    
            l2.s1 = True
        else: 
            l2.s2 = True
        
        # if (l1,l2) in self.internalLinks or (l2,l1) in self.internalLinks:
        #     return True
        b = False
        for _ in range(times):
            b = random.random() <= self.q
            if b:
                break
        if b:
            self.internalLinks.append((l1, l2))
        return b



    def attemptSwapping2(self, l1, l2 , times = 1 , timeSlot = 0):  # l1 -> Link, l2 -> Link
        # print('#####################################################time:' , timeSlot , 'node:', self.id , 'len:', len(self.prevInternalLinks))
        b = False
        for _ in range(times):
            b = random.random() <= self.q
            if b:
                break
        if (l1,l2) in self.prevInternalLinks or (l2,l1) in self.prevInternalLinks:
            # print('************************###time:' , timeSlot , 'link inside')

            if l1.isEntangled(timeSlot) and l2.isEntangled(timeSlot):
                # print('************************###time:' , timeSlot , 'TRUE!!!!!!')
                b = True
        if b:
            if l1.n1 == self:    
                l1.s1 = True
            else:       
                l1.s2 = True
            
            if l2.n1 == self:    
                l2.s1 = True
            else: 
                l2.s2 = True
            self.internalLinks.append((l1, l2))
        return b

    def assignIntermediate(self): # for intermediate 
        self.remainingQubits -= 1

    def clearIntermediate(self):
        self.remainingQubits += 1
        