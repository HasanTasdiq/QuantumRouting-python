import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
import sys
sys.path.append("..")
from quantum.topo.helper import needlink_timeslot
import math
ENTANGLEMENT_LIFETIME = 10
	
class RoutingEnv(Env):
    def __init__(self , algo):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = algo
        self.shower_length = 60
        self.algo = algo
        self.SIZE = len(algo.topo.nodes)
        self.OBSERVATION_SPACE_VALUES = (self.SIZE *2 +1,self.SIZE,)  
        self.ACTION_SPACE_SIZE = 2
        print('=========in Routing env ===== ' , algo.name)

    def step(self, pair ,  action , timeSlot):
        reward = 0
        state = None
        if action:
            count = self.algo.tryPreSwapp_rl(pair)
            # if count:
            #     print("+++++++============+++++++++ count of pre swapped links " , count)
            state = self.pair_state(pair , timeSlot)

        # Setting the placeholder for info
        info = {}
        
        # Returning the step information
        return state
    def find_reward(self , pair , action , timeSlot):
        reward = 0
        n1 = pair[0]
        n2 = pair[1]
        if (pair[0] , pair[1] , timeSlot) in self.algo.topo.reward:
            reward = self.algo.topo.reward[(pair[0] , pair[1] , timeSlot)]
        elif (pair[1] , pair[0] , timeSlot) in self.algo.topo.reward:
            reward = self.algo.topo.reward[(pair[1] , pair[0] , timeSlot)]
        elif self.algo.timeSlot - timeSlot >= ENTANGLEMENT_LIFETIME:
            reward = -10
        if not action:
            return -reward
        return reward




    def reset(self):
        self.state = self.algo.topo.needLinksDict
        self.shower_length = 60 
        return self.state
    def pair_state(self , pair , timeSlot):
        state1 = [0] * self.SIZE
        source = pair[0]
        dest = pair[1]
        count = 0
        # print(self.algo.topo.needLinksDict)
        # if (source , dest) in self.algo.topo.needLinksDict:
        #     timesUsed = len(self.algo.topo.needLinksDict[(source , dest)])
        # else:
        #     timesUsed = len(self.algo.topo.needLinksDict[(dest , source)])

        # try:
        #     timesUsed = len(self.algo.topo.needLinksDict[(source , dest)])
        #     # count = self.algo.topo.virtualLinkCount[(source , dest)]
        # except:
        #     timesUsed = len(self.algo.topo.needLinksDict[(dest , source)])
        #     # count = self.algo.topo.virtualLinkCount[(dest , source)]


        # demand = math.ceil(timesUsed  / needlink_timeslot) - count
        state1[source.id] = 1
        state1[dest.id] = 1

        graph_state = [[0]*self.SIZE]*self.SIZE

        for link in self.algo.topo.links:
            if link.isEntangled(timeSlot):
                n1 = link.n1.id
                n2 = link.n2.id
                graph_state[n1][n2] += 1
                graph_state[n2][n1] += 1
        # print(state1)
        # print(pair)
        graph_state.append(state1)

        req_state = [[0]*self.SIZE]*self.SIZE
        if  hasattr(self.algo , 'requestState' ):
            for req in self.algo.requestState:
                n1 = req[0].id 
                n2 = req[1].id 
                req_state[n1][n2] += 1
                req_state[n2][n1] += 1
            graph_state.extend(req_state)
        elif hasattr(self.algo , 'requests' ):
            for req in self.algo.requests:
                n1 = req[0].id 
                n2 = req[1].id 
                req_state[n1][n2] += 1
                req_state[n2][n1] += 1
            graph_state.extend(req_state)

        return np.array(graph_state)
    

        #     timesUsed = len(self.topo.needLinksDict[(source , dest)])
        # if timesUsed <= needlink_timeslot * self.topo.preSwapFraction:
        #     return 0
        # k = self.topo.hopsAway2(source , dest , 'Hop') - 1
            
        # if self.topo.virtualLinkCount[(source , dest)] * k >= math.ceil(timesUsed  / needlink_timeslot):
        #     return 0