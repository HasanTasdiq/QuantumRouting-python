import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
	
class RoutingEnv(Env):
    def __init__(self , algo):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = algo
        self.shower_length = 60
        self.algo = algo
        self.SIZE = len(algo.topo.nodes)
        self.OBSERVATION_SPACE_VALUES = (self.SIZE +1,self.SIZE,)  
        self.ACTION_SPACE_SIZE = 2
        print('=========in Routing env ===== ' , algo.name)

    def step(self, pair ,  action):
        reward = 0
        if action:
            reward = self.algo.tryPreSwapp_rl(pair)

        # Setting the placeholder for info
        info = {}
        
        # Returning the step information
        return self.state, reward, info
    
    def reset(self):
        self.state = self.algo.topo.needLinksDict
        self.shower_length = 60 
        return self.state
    def pair_state(self , pair , timeSlot):
        state1 = [0] * self.SIZE
        state1[pair[0]] = 1
        state1[pair[1]] = 1

        graph_state = [[0]*self.SIZE]*self.SIZE

        for link in self.algo.topo.links:
            if link.isEntangled(timeSlot):
                n1 = link.n1.id
                n2 = link.n2.id
                graph_state[n1][n2] += 1
                graph_state[n2][n1] += 1
        
        graph_state.append(state1)

        return np.array(graph_state)