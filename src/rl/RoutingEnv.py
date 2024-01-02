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

    def step(self, action):
        self.state += action -1 
        self.shower_length -= 1 
        
        # Calculating the reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Checking if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Setting the placeholder for info
        info = {}
        
        # Returning the step information
        return self.state, reward, done, info
    
    def reset(self):
        self.state = self.algo
        self.shower_length = 60 
        return self.state