
from .Node import Node
import random
import math

class Link:
    
    def __init__(self, topo, n1: Node, n2: Node, s1: bool, s2: bool, id: int, l: float):
        self.id, self.n1, self.n2, self.s1, self.s2, self.alpha = id, n1, n2, s1, s2, topo.alpha
        self.assigned = False
        self.entangled = False
        self.entangledTimeSlot = 0
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
    def isEntangled(self , timeSlot):
        return self.entangled and (timeSlot - self.entangledTimeSlot) < 10



    def assignQubits(self):
        # prevState = self.assigned
        self.assigned = True
        self.n1.remainingQubits -= 1
        self.n2.remainingQubits -= 1
  
    def clearEntanglement(self):
        preState = self.assigned
        self.s1 = False
        self.s2 = False
        self.assigned = False
        self.entangled = False

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)

        if preState:
            self.n1.remainingQubits += 1
            self.n2.remainingQubits += 1
    
    def keepEntanglementOnly(self):
        preState = self.assigned
        self.s1 = False
        self.s2 = False
        self.assigned = False
        # self.entangled = False

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)

        if preState:
            self.n1.remainingQubits += 1
            self.n2.remainingQubits += 1
    
    def clearPhase4Swap(self):
        self.s1 = False
        self.s2 = False
        self.entangled = False

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)
                        
    def keepPhase4Swap(self):
        self.s1 = False
        self.s2 = False
        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)
                # self.n1.prevInternalLinks.append((internalLink[0] , internalLink[1] , self.entangledTimeSlot))
                self.n1.prevInternalLinks.append((internalLink[0] , internalLink[1] ))

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)
                        # self.n2.prevInternalLinks.append((internalLink[0] , internalLink[1] , self.entangledTimeSlot))
                        self.n2.prevInternalLinks.append((internalLink[0] , internalLink[1] ))
        
         
    # def tryEntanglement(self , timeSlot = 0):
    #     # print('ent prob', self.p)

    #     b = self.assigned and self.p >= random.random()
    #     if b:
    #         self.entangledTimeSlot = timeSlot
    #         self.entangled = b
            
    #     if self.entangled and (timeSlot - self.entangledTimeSlot) < 10:
    #         return True
    #     return b

    def tryEntanglement3(self , timeSlot = 0):
        # print('ent prob', self.p)

        b = self.assigned and self.p >= random.random()
        if b:
            self.entangledTimeSlot = timeSlot
            self.entangled = b
            
        if self.entangled and (timeSlot - self.entangledTimeSlot) < 10:
            return True
        return b


    def tryEntanglement2(self , timeSlot = 0):
        # print('ent prob', self.p)
        # print('+++++++++++++++++++++++ every')
        b = self.p >= random.random()
        if b:
            self.entangledTimeSlot = timeSlot
            self.entangled = b
            
        if self.entangled and (timeSlot - self.entangledTimeSlot) < 10:
            # print('tryEntanglement2 ' , self.n1.id , self.n2.id , timeSlot , True)
            return True

        # print('tryEntanglement2 ', self.n1.id , self.n2.id , timeSlot , b)
        return b

    def tryEntanglement1(self , timeSlot = 0):
        # print('ent prob', self.p)
        # if self.entangled:
        #     if (timeSlot - self.entangledTimeSlot) >= 10:
        #         print('!!!!! ++++ ==== tryEntanglement1 (timeSlot - self.entangledTimeSlot) >= 10', self.n1.id , self.n2.id , timeSlot , True)

        if self.entangled and (timeSlot - self.entangledTimeSlot) < 10:
            # print('tryEntanglement1 ', self.n1.id , self.n2.id , timeSlot , True)
            return True
        # b = self.assigned and self.p >= random.random()
        b = self.p >= random.random()
        if b:
            self.entangledTimeSlot = timeSlot
            self.entangled = b
            
        # print('tryEntanglement1 ', self.n1.id , self.n2.id , timeSlot , b)
        return b
    
    def tryEntanglement(self , timeSlot = 0, param=None):
        if param == 'every':
            return self.tryEntanglement2(timeSlot)
        elif param == 'ten' or param == 'everya':
            return self.tryEntanglement1(timeSlot)
        else:
            return self.tryEntanglement3(timeSlot)

    
  
    def assignable(self): 
        return not self.assigned and self.n1.remainingQubits > 0 and self.n2.remainingQubits > 0
