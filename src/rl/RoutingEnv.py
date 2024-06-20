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
    def assignQubit(self, link , action, timeSlot):
        state = None
        if action and link.assignable():
            link.assignQubits()
            state = self.ent_state(link , timeSlot)
        return state
    def assignQubitEdge(self, edge , action, timeSlot):
        state = None
        if action:
            for link in edge[0].links:
                if link.n2 == edge[1] and link.assignable():
                    if np.random.random() > 0.5:
                        link.assignQubits()
            state = self.ent_state(edge , timeSlot)
        return state   
     
    # def assignQubitEdge(self, edge , action, timeSlot):
    #     state = None
    #     links = []
    #     for l in edge[0].links:
    #         if l.n2 == edge[1]:
    #             links.append(l)

    #     for i in range(action):
    #         link = links[i]
    #         if link.assignable():
    #             link.assignQubits()
    #         state = self.ent_state(edge , timeSlot)
    #     return state
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

    def find_reward_ent(self , link  , timeSlot , action):
        reward = 0
        edge = link
        if link[1].id < link[0].id:
            edge = (link[1] , link[0])

        if edge in self.algo.topo.reward_ent:
            reward = self.algo.topo.reward_ent[edge]
            if not action:
                # reward = self.algo.topo.negative_reward if reward == self.algo.topo.positive_reward else self.algo.topo.positive_reward
                reward = -reward
        elif self.algo.timeSlot - timeSlot >= ENTANGLEMENT_LIFETIME:
            reward = -10
        return reward

    def reset(self):
        self.state = self.algo.topo.needLinksDict
        self.shower_length = 60 
        return self.state
    
    def getedges(self , a):
        res = []
        for i in range(len(a) - 1):
            res.append((a[i] , a[i + 1]))
        return res
    def ent_state(self, link , timeSlot):
        state1 = [0 for i in range(self.SIZE)]
        state_qm = [0 for i in range(self.SIZE)]

        try:
            source = link.n1
            dest = link.n2
        except:
            source = link[0]
            dest = link[1]

        count = 0
        state1[source.id] = 1
        state1[dest.id] = 1

        graph_state = [[0]*self.SIZE]*self.SIZE

        for link in self.algo.topo.links:
            if link.assignable():
                n1 = link.n1.id
                n2 = link.n2.id
                graph_state[n1][n2] += 1
                graph_state[n2][n1] += 1
        for node in self.algo.topo.nodes:
            state_qm[node.id] = node.remainingQubits
        # print(state1)
        # print(pair)

        req_state = [[0]*self.SIZE]*self.SIZE
        # if  hasattr(self.algo , 'requestState' ):
        #     for req in self.algo.requestState:
        #         n1 = req[0].id 
        #         n2 = req[1].id 
        #         req_state[n1][n2] += 1
        #         req_state[n2][n1] += 1
        #     graph_state.extend(req_state)
        # elif hasattr(self.algo , 'requests' ):
        #     for req in self.algo.requests:
        #         n1 = req[0].id 
        #         n2 = req[1].id 
        #         req_state[n1][n2] += 1
        #         req_state[n2][n1] += 1
        #     graph_state.extend(req_state)


        if  hasattr(self.algo , 'requestState' ):
            for req in self.algo.requestState:
                n1,n2 = min(req[0].id ,req[1].id ) , max(req[0].id ,req[1].id ) 
                print('---------------------------------==-=-=-=-=-=-=-=-=-=--')
                for path , l in self.algo.topo.pair_edge_dict[(n1 , n2)]:
                    print(path)
                req_state[n1][n2] += 1
                req_state[n2][n1] += 1
            graph_state.extend(req_state)
        elif hasattr(self.algo , 'requests' ):
            for req in self.algo.requests:
                n1,n2 = min(req[0].id ,req[1].id ) , max(req[0].id ,req[1].id ) 
                # print('---------------------------------==-=-=-=-=-=-=-=-=-=--')
                for path, l  in self.algo.topo.pair_edge_dict[(n1 , n2)]:
                    # print(path)
                    for pair in self.getedges(path):
                        # print(pair)
                        req_state[pair[0]][pair[1]] += 1
                        req_state[pair[1]][pair[0]] += 1
            graph_state.extend(req_state)
            
        graph_state.append(state1)
        graph_state.append(state_qm)


        return np.array(graph_state)
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