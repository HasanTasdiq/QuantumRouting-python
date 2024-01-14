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