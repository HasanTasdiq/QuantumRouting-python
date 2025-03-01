import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
import sys
sys.path.append("..")
# from quantum.topo.helper import needlink_timeslot
import math
import ast
import time

ENTANGLEMENT_LIFETIME = 10

import tensorflow as tf
from keras.layers import Embedding, Flatten, Attention
import numpy
from objsize import get_deep_size


	
class RoutingEnv(Env):
    def __init__(self , algo):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = algo
        self.shower_length = 60
        self.algo = algo
        self.SIZE = len(algo.topo.nodes)
        self.ACTION_SPACE_SIZE = 2

        self.graph = []
        self.V = self.SIZE
        self.embedding_layer = Embedding(input_dim=20, output_dim=1)
        self.attention_layer = Attention()

        self.skip = True


        

        # print('=========in Routing env ===== ' , algo.name , 'self size ' , get_deep_size(self)/1000000)

    def step(self, pair ,  action , timeSlot):
        reward = 0
        state = None
        if action:
            count = self.algo.tryPreSwapp_rl(pair)
            if count:
                print("+++++++============+++++++++ count of pre swapped links " , count)
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
    def assignQubitEdge_2(self, edge , action, timeSlot):
        state = None
        assignable = False
        if action:
            for link in edge[0].links:
                if link.n2 == edge[1] and link.assignable():
                    if np.random.random() > 0.5:
                        link.assignQubits()
                        self.algo.totalUsedQubits += 2
                        assignable = True
            state = self.ent_state(edge , timeSlot)
        return state , assignable   
     
    def assignQubitEdge(self, edge , action, timeSlot):
        state = None
        links = []
        assignable = False

        for l in edge[0].links:
            if l.n2 == edge[1] and l.assignable():
                links.append(l)

        for i in range(min(action , len(links))):
            link = links[i]
            if link.assignable():
                link.assignQubits()
                self.algo.totalUsedQubits += 2

                assignable = True
            state = self.ent_state(edge , timeSlot)
        return state , assignable
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
    
    def find_reward_routing(self, request , timeSlot,current_node_id,  action):
        reward = 0
        # print('------------find_reward_routing---------' , current_node_id)
        key = str(request[0].id) + '_' + str(request[1].id) + '_' + str(current_node_id) + '_' + str(action)
        reward  = self.algo.topo.reward_routing[key]

        # if (request,current_node_id , action) in self.algo.topo.reward_routing:
        #     print('------------find_reward_routing---------')
        #     key = str(request[0].id) + '_' + str(request[1].id) + '_' + current_id + '_' + action.id
        #     reward = self.algotopo.reward_routing[key]
        # else:
        #     reward = -10
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
    def minDistance(self, dist, sptSet):

        # Initialize minimum distance for next node
        min = sys.maxsize
        min = 100
        min_index = -1

        # Search not nearest vertex not in the
        # shortest path tree
        for u in range(self.V):
            if dist[u] < min and sptSet[u] == False:
                min = dist[u]
                min_index = u

        return min_index
    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])
    def dijkstra(self, src):

        # dist = [sys.maxsize] * self.V
        dist = [100] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # x is always equal to src in first iteration
            x = self.minDistance(dist, sptSet)

            if x < 0:
                continue

            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[x] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for y in range(self.V):
                if self.graph[x][y] > 0 and sptSet[y] == False and \
                        dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]

        # self.printSolution(dist)
        return dist
    def get_embedded_output(self , inp , formatted = False):

        # Apply the Embedding layer
        t = time.time()
        embedded_output = self.embedding_layer(tf.constant(inp))
        # print('self.embedding_layer(tf.constant(inp)) ' , time.time()-t)

        # # Flatten the output
        # t = time.time()

        # flattened_output = Flatten()(embedded_output)


        # print('Flatten()(embedded_output) ' , time.time()-t)


        # # Print the shape of the flattened output
        # # print(embedded_output)
        # # print(flattened_output)
        # t = time.time()

        # out1 = Flatten()(tf.constant([numpy.array(flattened_output)]))[0]
        # print('Flatten()(tf.constant([numpy.array(flattened_output)]) ' , time.time()-t)

        out1 = embedded_output
        if formatted:
            return list(out1.numpy())
        return out1
    
    def get_emb_attention(self , inp):
        atten_input = self.get_embedded_output(inp)
        input_vector = tf.constant(atten_input, dtype=tf.float32)

        # Expand dimensions to simulate a sequence
        # Shape: (batch_size=1, seq_length=1, embedding_dim=6)
        # input_vector = tf.expand_dims(tf.expand_dims(input_vector, axis=0), axis=0)

        # Create query, key, and value as the same (self-attention)
        query = key = value = input_vector  # Shape: (batch_size=1, seq_length=1, embedding_dim=6)

        # Initialize the Attention layer

        # Compute attention
        attention_output = self.attention_layer([query, key, value])

        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(attention_output)
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        return attention_output.numpy()
    def schedule_state_prob(self , requests):
        state_qubits = [0 for i in range(self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        S = []
        U = []
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(len(requests))
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')

        for req in requests:
            state_req = [0 for i in range(2*self.SIZE)]

            if not req[2]:
                state_req[req[0].id] = 1
                state_req[self.SIZE+req[1].id] = 1
            # usd = self.get_embedded_output(state_req , formatted=True)
            # asd = self.get_emb_attention(state_req)

            # usd.extend(asd)

            U.append(state_req)
        Usd = self.get_embedded_output(U , formatted=True)
        Asd = self.get_emb_attention(U)

        # print('----------------------')
        # print('----------------------')
        # print('---------asd---------')
        # print(Asd)
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        for usd in Usd:
            tmp = []
            for el in usd:
                tmp.append(el[0])
            S.append(tmp)
        for i in range(len(Asd)):
            asd = Asd[i]
            tmp = []
            for el in asd:
                tmp.append(el[0])
            S[i].extend(tmp)
        

        # print(S)



        

        
        for link in self.algo.topo.links:
            if link.assignable():
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] = 1
                state_graph[n2][n1] = 1
        for u in S:
            a = self.get_embedded_output(state_graph )
            a = Flatten()(tf.constant([numpy.array(a)]))[0]
            a = list(a.numpy())
            

            # print('----------------------')
            # print(Asd)
            # print('----------------------')        
            # print('----------------------')
            # print('----------------------')
            # print('----------------------')


            u.extend(a)
        for node in self.algo.topo.nodes:
            state_qubits[node.id] = node.remainingQubits
        
        for u in S:
            c = self.get_embedded_output(state_qubits)
            c = Flatten()(tf.constant([numpy.array(c)]))[0]
            c = list(c.numpy())

            u.extend(c)
        
        # print('----------------------')
        # print('----------------------')
        # print('-----------S-----------')
        # print(S)
        # print('----------------------')
        # print('----------------------')
        return np.array(S)
        
    def schedule_state(self , requests):
        state_qubits = [0 for i in range(self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        
        S = []
        U = []
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(len(requests))
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')

        for req in requests:
            state_req = [0 for i in range(2*self.SIZE)]

            if not req[2]:
                state_req[req[0].id] = 1
                state_req[self.SIZE+req[1].id] = 1
            # usd = self.get_embedded_output(state_req , formatted=True)
            # asd = self.get_emb_attention(state_req)

            # usd.extend(asd)

            U.append(state_req)
        Usd = self.get_embedded_output(U , formatted=True)
        Asd = self.get_emb_attention(U)

        # print('----------------------')
        # print('----------------------')
        # print('---------asd---------')
        # print(Asd)
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        for usd in Usd:
            tmp = []
            for el in usd:
                tmp.append(el[0])
            S.append(tmp)
        for i in range(len(Asd)):
            asd = Asd[i]
            tmp = []
            for el in asd:
                tmp.append(el[0])
            S[i].extend(tmp)
        

        # print(S)



        

        
        for link in self.algo.topo.links:
            if link.assignable():
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] = 1
                state_graph[n2][n1] = 1
                if 'prob' in self.algo.name:
                    state_graph[n1][n2] = link.p()
                    state_graph[n2][n1] = link.p()
        for u in S:
            a = self.get_embedded_output(state_graph )
            a = Flatten()(tf.constant([numpy.array(a)]))[0]
            a = list(a.numpy())
            

            # print('----------------------')
            # print(Asd)
            # print('----------------------')        
            # print('----------------------')
            # print('----------------------')
            # print('----------------------')


            u.extend(a)
        for node in self.algo.topo.nodes:
            state_qubits[node.id] = node.remainingQubits
        
        for u in S:
            c = self.get_embedded_output(state_qubits)
            c = Flatten()(tf.constant([numpy.array(c)]))[0]
            c = list(c.numpy())

            u.extend(c)
        
        # print('----------------------')
        # print('----------------------')
        # print('-----------S-----------')
        # print(S)
        # print('----------------------')
        # print('----------------------')
        return np.array(S)
    def schedule_route_state(self , requests):
        state_q = [0 for i in range(self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        S = []
        U = []
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(len(requests))
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')

        for req in requests:
            state_req = [0 for i in range(2*self.SIZE)]

            if not req[2]:
                state_req[req[0].id] = 1
                state_req[self.SIZE+req[1].id] = 1
            # usd = self.get_embedded_output(state_req , formatted=True)
            # asd = self.get_emb_attention(state_req)

            # usd.extend(asd)

            U.append(state_req)
        Usd = self.get_embedded_output(U , formatted=True)
        Asd = self.get_emb_attention(U)

        # print('----------------------')
        # print('----------------------')
        # print('---------asd---------')
        # print(Asd)
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        for usd in Usd:
            tmp = []
            for el in usd:
                tmp.append(el[0])
            S.append(tmp)
        for i in range(len(Asd)):
            asd = Asd[i]
            tmp = []
            for el in asd:
                tmp.append(el[0])
            S[i].extend(tmp)
        

        # print(S)



        

        
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] = 1
                state_graph[n2][n1] = 1
        for u in S:
            a = self.get_embedded_output(state_graph )
            a = Flatten()(tf.constant([numpy.array(a)]))[0]
            a = list(a.numpy())
            

            # print('----------------------')
            # print(Asd)
            # print('----------------------')        
            # print('----------------------')
            # print('----------------------')
            # print('----------------------')


            u.extend(a)
        for node in self.algo.topo.nodes:
            state_q[node.id] = node.q
        
        for u in S:
            c = self.get_embedded_output(state_q)
            c = Flatten()(tf.constant([numpy.array(c)]))[0]
            c = list(c.numpy())

            u.extend(c)
        
        # print('----------------------')
        # print('----------------------')
        # print('-----------S-----------')
        # print(S)
        # print('----------------------')
        # print('----------------------')
        return np.array(S)
        
        
        

            
        # for self.


    #with schedule and routing 
    def schedule_routing_state_long(self , requests=None):
        state_q = [0 for i in range(self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        S = []
        U = []
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] += 1
                state_graph[n2][n1] += 1

        if requests is None:
            requests = self.algo.requestState

        for req in requests:
            state_req = [0 for i in range(2*self.SIZE)]
            neighbors = [0 for i in range(self.SIZE)]

            if not req[5]:
                state_req[req[2].id] = 1 #current node
                state_req[self.SIZE+req[1].id] = 1
            else:
                state_req[req[1].id] = 1 #current node
                state_req[self.SIZE+req[1].id] = 1
                # neighbors = state_graph[req[2].id]
            # state_req.extend(neighbors)

            U.append(state_req)
        # if self.algo.timeSlot%100 == 0:
        #     print('------------------state req ------------')
        #     for req in U:
        #         print(req)
        t = time.time()
        Usd = self.get_embedded_output(U , formatted=True)
        # print('Usd = self.get_embedded_output ' , time.time()-t)
        t = time.time()

        Asd = self.get_emb_attention(U)
        # print('asd = self.get_embedded_output ' , time.time()-t)


        for usd in Usd:
            tmp = []
            for el in usd:
                tmp.append(el[0])
            S.append(tmp)
        for i in range(len(Asd)):
            asd = Asd[i]
            tmp = []
            for el in asd:
                tmp.append(el[0])
            S[i].extend(tmp)
        
        t = time.time()

        a = self.get_embedded_output(state_graph )
        # print('a = self.get_embedded_outpu ' , time.time()-t)
        
        a = Flatten()(tf.constant([numpy.array(a)]))[0]
        a = list(a.numpy())
        for u in S:
            u.extend(a)
            
        # for node in self.algo.topo.nodes:
        #     state_q[node.id] = node.q
        
        # for u in S:
        #     c = self.get_embedded_output(state_q)
        #     c = Flatten()(tf.constant([numpy.array(c)]))[0]
        #     c = list(c.numpy())

        #     u.extend(c)
        # if self.algo.timeSlot%100 == 0:
        #     for u in S:
        #         print(u)
        
        return np.array(S)
    def schedule_routing_state(self , requests=None):
        state_q = [0 for i in range(self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        state_dist = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        S = []
        U = []
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] += 1
                state_graph[n2][n1] += 1

        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_dist[n1][n2] = link.l / 200
                state_dist[n2][n1] = link.l / 200

        if requests is None:
            requests = self.algo.requestState

        for req in requests:
            state_req = [0 for i in range(self.SIZE)]
            neighbors = [0 for i in range(self.SIZE)]

            if not req[5]:
                state_req[req[2].id] = 1 #current node
                state_req[req[1].id] = 10
                # neighbors = state_graph[req[2].id]
            # state_req.extend(neighbors)

            U.append(state_req)
        # if self.algo.timeSlot%100 == 0:
        #     print('------------------state req ------------')
        #     for req in U:
        #         print(req)
        t = time.time()
        # Usd = self.get_embedded_output(U , formatted=True)
        # print('Usd = self.get_embedded_output ' , time.time()-t)
        t = time.time()

        U.extend(state_graph)
        U.extend(state_dist)

        Asd = self.get_emb_attention(U)
        # print('asd = self.get_embedded_output ' , time.time()-t)


        for asd in Asd:
            tmp = []
            for el in asd:
                tmp.append(el[0])
            S.append(tmp)
        # for i in range(len(Asd)):
        #     asd = Asd[i]
        #     tmp = []
        #     for el in asd:
        #         tmp.append(el[0])
        #     S[i].extend(tmp)
        
        t = time.time()

        # a = self.get_embedded_output(state_graph )
        # for r in a:
        #     tmp = []
        #     for el in r:
        #         tmp.append(el[0])
        #     S.append(tmp)
        # print('aaaaaa  ')
        # print(S)
        # S.extend(a)

        # print('a = self.get_embedded_outpu ' , time.time()-t)
        
        # a = Flatten()(tf.constant([numpy.array(a)]))[0]
        # a = list(a.numpy())
        # for u in S:
        #     u.extend(a)
            
        # for node in self.algo.topo.nodes:
        #     state_q[node.id] = node.q
        
        # for u in S:
        #     c = self.get_embedded_output(state_q)
        #     c = Flatten()(tf.constant([numpy.array(c)]))[0]
        #     c = list(c.numpy())

        #     u.extend(c)
        # if self.algo.timeSlot%100 == 0:
        #     for u in S:
        #         print(u)
        
        return np.array(S)
    
    def routing_state_(self , current_request , current_node_id,path, timeSlot = 0  ):
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        state_cr = [0 for i in range(self.SIZE)]
        state_cn = [0 for i in range(self.SIZE)]



        if current_node_id == current_request[1].id or current_node_id == self.SIZE:
            # print('len req state ::: ' , len(self.algo.requestState) , current_node_id)
            for  i , req in enumerate(self.algo.requestState):
                if req[0].id == current_request[0].id and req[1].id == current_request[1].id:
                    # print('iiiiiiiiiiiiiii ' , i , (current_request[0].id , current_request[1].id))
                    current_request = (self.algo.requestState[i+1][0] , self.algo.requestState[i+1][1])
                    current_node_id = current_request[0].id
                    # print('state for next req', (current_request[0].id , current_request[1].id))
                    break
        
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] += 1
                state_graph[n2][n1] += 1

        state_cr[current_request[0].id] = 1
        state_cr[current_request[1].id] = 1

        # state_cn[current_node_id] = 1
        state_cn = state_graph[current_node_id]

        S = []
        U = []
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(len(requests))
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')

        RS = []
        r = 0 
        for req in self.algo.requestState:
            state_req = [0 for i in range(self.SIZE)]

            if not req[5]:
                
                src_id = req[0].id
                dst_id = req[1].id
                curr_id = req[2].id

                # reqStates[r][src_id] = 1
                state_req[dst_id] = 1
                state_req[src_id] = 1
            U.append(state_cn + state_cr )
            # U.append(state_cn + state_cr + state_req)
            RS.append(state_req)


        
        # Usd = self.get_embedded_output(U , formatted=True)
        Asd = self.get_emb_attention(U)

        Usd = self.get_embedded_output(RS , formatted=True)


        # print('----------------------')
        # print('----------------------')
        # print('---------asd---------')
        # print(Asd)
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        for usd in Asd:
            tmp = []
            for el in usd:
                tmp.append(el[0])
            S.append(tmp)
        
        for i in range(len(Usd)):
            usd = Usd[i]
            tmp = []
            for el in usd:
                tmp.append(el[0])
            S[i].extend(tmp)

        # print('----------------------')
        # print('----------------------')
        # print('-----------S-----------')
        # print(S)
        # print('----------------------')
        # print('----------------------')
        # for i in range(len(Asd)):
        #     asd = Asd[i]
        #     tmp = []
        #     for el in asd:
        #         tmp.append(el[0])
        #     S[i].extend(tmp)
        

        # print(S)



        

        

        for u in S:
            a = self.get_embedded_output(state_graph )
            a = Flatten()(tf.constant([numpy.array(a)]))[0]
            a = list(a.numpy())
            

            # print('----------------------')
            # print(Asd)
            # print('----------------------')        
            # print('----------------------')
            # print('----------------------')
            # print('----------------------')


            u.extend(a)
        
        # print('----------------------')
        # print('----------------------')
        # print('-----------S-----------')
        # print(S)
        # print('----------------------')
        # print('----------------------')
        return np.array(S)
    def get_next_current_node_id(self ,current_request , current_node_id ):
        if current_node_id == current_request[1].id or current_node_id == self.SIZE:
            # print('len req state ::: ' , len(self.algo.requestState) , current_node_id)
            for  i , req in enumerate(self.algo.requestState):
                if req[0].id == current_request[0].id and req[1].id == current_request[1].id:
                    # print('iiiiiiiiiiiiiii ' , i , (current_request[0].id , current_request[1].id))
                    current_request = (self.algo.requestState[i+1][0] , self.algo.requestState[i+1][1])
                    current_node_id = current_request[0].id
                    # print('state for next req', (current_request[0].id , current_request[1].id))
                    break
        return current_node_id
    
    def routing_state2(self , current_request , current_node_id ,path, timeSlot = 0  ):
        if current_node_id == current_request[1].id or current_node_id == self.SIZE:
            # print('len req state ::: ' , len(self.algo.requestState) , current_node_id)
            for  i , req in enumerate(self.algo.requestState):
                if req[0].id == current_request[0].id and req[1].id == current_request[1].id:
                    # print('iiiiiiiiiiiiiii ' , i , (current_request[0].id , current_request[1].id))
                    current_request = (self.algo.requestState[i+1][0] , self.algo.requestState[i+1][1])
                    current_node_id = current_request[0].id
                    # print('state for next req', (current_request[0].id , current_request[1].id))
                    break



        state_cr = [0 for i in range(self.SIZE)]

        state_cn = [0 for i in range(self.SIZE)]

        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]

        reqStates = [[0 for column in range(self.SIZE)]
                      for row in range(self.algo.topo.numOfRequestPerRound)]



        # print(len(self.algo.topo.links))
        count =0
        for link in self.algo.topo.links:
            if link.isEntangled(timeSlot) and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] += 1
                state_graph[n2][n1] += 1
                count += 1

        # print(state_graph)
        
        state_graph = self.get_embedded_output(state_graph , formatted=True)
        state_graph = Flatten()(tf.constant([numpy.array(state_graph)]))[0]
        state_graph = list(state_graph.numpy())
        # print('--------')
        # print('--------')
        # print('--------')
        # print('--------')
        # print(state_graph)
        # print('--------')
        # print('--------')
        # print('--------')

        # S = []
        # for r in state_graph:
        #     tmp = []
        #     for r2 in r:
        #         tmp.append(r2[0])
        #     S.append(tmp)
        # # state_graph = self.embedding_layer(state_graph)
        # state_graph = S
        # print(state_graph)
        
        state_cr[current_request[0].id] = 1
        state_cr[current_request[1].id] = 1

        # print(state_cr)
        state_cr = self.get_emb_attention(state_cr)
        S = []
        for r in state_cr:
            S.append(r[0])
        state_cr = S
        # print(state_cr)

        # print('555555555555555')
        try:
            state_cn[current_node_id] = 1
        except:
            p = 0 #nothing

        # print(state_cn)

        state_cn = self.get_emb_attention(state_cn)
        S = []
        for r in state_cn:
            S.append(r[0])
        state_cn = S


        r = 0 
        for req in self.algo.requestState:
            if req[5]:
                continue
            src_id = req[0].id
            dst_id = req[1].id
            curr_id = req[2].id

            # reqStates[r][src_id] = 1
            reqStates[r][dst_id] = 1
            reqStates[r][curr_id] = 1

            r += 1
        
        # state_req = self.get_embedded_output(state_req , formatted=True)
        # print(reqStates)
        reqStates = np.array(self.get_embedded_output(reqStates))
        S = []
        for r in reqStates:
            tmp = []
            for r2 in r:
                tmp.append(r2[0])
            S.append(tmp)
        reqStates = S
        # print(reqStates)

        ret = []

        r= 0
        for req in self.algo.requestState:
            src_id = req[0].id
            dst_id = req[1].id
            curr_id = req[2].id
            curr_node = [0 for _ in range(self.SIZE)]
            if current_request[0].id == src_id and current_request[1].id == dst_id:
                curr_node = state_cn
            
            req = reqStates[r]

            curr_node.extend(state_cr)
            curr_node.extend(req)
            curr_node.extend(state_graph)
            ret.append(curr_node)

            r +=1








        # state_graph.append(state_cr)
        # state_graph.append(state_cn)
        # state_graph.extend(reqStates)

        return np.array(ret)

    def routing_state(self , current_request , current_node_id ,path, timeSlot = 0  ):
        if current_node_id == current_request[1].id or current_node_id == self.SIZE:
            # print('len req state ::: ' , len(self.algo.requestState) , current_node_id)
            for  i , req in enumerate(self.algo.requestState):
                if req[0].id == current_request[0].id and req[1].id == current_request[1].id:
                    # print('iiiiiiiiiiiiiii ' , i , (current_request[0].id , current_request[1].id))
                    current_request = (self.algo.requestState[i+1][0] , self.algo.requestState[i+1][1])
                    current_node_id = current_request[0].id
                    # print('state for next req', (current_request[0].id , current_request[1].id))
                    break



        state_cr = [0 for i in range(self.SIZE)]
        state_cn = [0 for i in range(self.SIZE)]
        # state_path = [0 for i in range(self.SIZE)]

        state_req = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        reqStates = [[0 for column in range(self.SIZE)]
                      for row in range(self.algo.topo.numOfRequestPerRound)]

        count =0
        for link in self.algo.topo.links:
            if link.isEntangled(timeSlot) and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] += 1
                state_graph[n2][n1] += 1
                count += 1

                # print(state_graph)
                
                graph[n1][n2] = 1
                graph[n2][n1] = 1
        # print(state_graph)


        # get from the graph
        state_cn = state_graph[current_node_id]
        # try:
        #     state_cn[current_node_id] = 1
        # except:
        #     p = 0 #nothing
        
        state_graph = self.get_embedded_output(state_graph , formatted=True)
        S = []
        for r in state_graph:
            tmp = []
            for r2 in r:
                tmp.append(r2[0])
            S.append(tmp)
        # state_graph = self.embedding_layer(state_graph)
        state_graph = S
        
        self.graph = graph
        # print(self.graph)

      
        state_cr[current_request[0].id] = 1
        state_cr[current_request[1].id] = 1

        # print(state_cr)
        state_cr = self.get_emb_attention(state_cr)
        S = []
        for r in state_cr:
            S.append(r[0])
        state_cr = S



        # print(state_cn)

        state_cn = self.get_emb_attention(state_cn)
        S = []
        for r in state_cn:
            S.append(r[0])
        state_cn = S


        r = 0 
        for req in self.algo.requestState:
            if req[5]:
                continue
            src_id = req[0].id
            dst_id = req[1].id
            curr_id = req[2].id

            # reqStates[r][src_id] = 1
            reqStates[r][dst_id] = 1
            reqStates[r][curr_id] = 1

            r += 1
        
        for req in self.algo.requestState:
            dst_id = req[1].id
            curr_id = req[2].id
            state_req[dst_id][curr_id] += 1
            state_req[curr_id][dst_id] += 1
        
        # state_req = self.get_embedded_output(state_req , formatted=True)
        # print(reqStates)
        reqStates = np.array(self.get_embedded_output(reqStates))
        S = []
        for r in reqStates:
            tmp = []
            for r2 in r:
                tmp.append(r2[0])
            S.append(tmp)
        reqStates = S
        # print(reqStates)





        state_graph.append(state_cr)
        state_graph.append(state_cn)
        # state_graph.append(dist)
        # state_graph.extend(state_req)
        state_graph.extend(reqStates)


        # state_graph.append(state_path)
        return np.array(state_graph)

        


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

        graph_state = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]

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

        req_state = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]


        if  hasattr(self.algo , 'requestState' ):
            for req in self.algo.requestState:
                n1,n2 = min(req[0].id ,req[1].id ) , max(req[0].id ,req[1].id ) 
                for path , l in self.algo.topo.pair_edge_dict[(n1 , n2)]:
                    for pair in self.getedges(path):
                        req_state[pair[0]][pair[1]] += 1
                        req_state[pair[1]][pair[0]] += 1
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
        

        dist_state = [[0]*self.SIZE]*self.SIZE

        for u in self.algo.topo.nodes:
            for v in self.algo.topo.nodes:
                dis = self.algo.topo.distance(u.loc, v.loc)
                dist_state[u.id][v.id] = dis
                dist_state[v.id][u.id] = dis
        graph_state.extend(dist_state)



        graph_state.append(state1)
        graph_state.append(state_qm)


        return np.array(graph_state)
    def pair_state(self , pair , timeSlot):
        state1 = [0 for column in range(self.SIZE)]
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

        graph_state = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]

        for link in self.algo.topo.links:
            if link.isEntangled(timeSlot):
                n1 = link.n1.id
                n2 = link.n2.id
                graph_state[n1][n2] += 1
                graph_state[n2][n1] += 1
        # print(state1)
        # print(pair)
        graph_state.append(state1)

        req_state = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
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
    
    # def neighbors(self , current_node , current_state):
    #     neighbor_list = current_state[current_node.id]

    #     ret = []

    #     for n in neighbor_list:
    #         if n > 0:
    #             ret.append(1)
    #         else:
    #             ret.append(0)
        
    #     return ret
    def get_neighbors(self , current_node_id):
        neighbor_list = [0 for _ in range(self.SIZE)]
        current_node = self.algo.topo.nodes[current_node_id]
        neighbors = [link.theOtherEndOf(current_node).id for link in current_node.links if link.isEntangled() and not link.taken]
        for n in neighbors:
            neighbor_list[n] = 1
        return neighbor_list
    
    def get_mask_shcedule_route_batch(self):
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] = 1
                state_graph[n2][n1] = 1
        

        mask = [None for _ in range(self.algo.topo.numOfRequestPerRound * self.SIZE)]
        for reqState in self.algo.requestState:

            src,dst,current_node , path , index , done = reqState
            # print('in get mask +++==== ' , src.id,dst.id,current_node.id , path , index , done)

            if done:
                continue
            neighbors = [i for i, x in enumerate(state_graph[current_node.id]) if x == 1]
            for n in neighbors:
                if n not in path and n != current_node.id:
                    mask[index*self.SIZE + n] = 1
        # print('get mask ' , mask)
        if not mask.count(1):
            mask = self.get_mask__request_shcedule_route()
        return mask
    def get_mask_one_req_schedule_route(self , reqState):
        mask = [None for _ in range(self.algo.topo.numOfRequestPerRound * self.SIZE)]
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] = 1
                state_graph[n2][n1] = 1
        src,dst,current_node , path , index , done = reqState

        neighbors = [i for i, x in enumerate(state_graph[current_node.id]) if x == 1]
        for n in neighbors:
            if n not in path and n != current_node.id:
                mask[index*self.SIZE + n] = 1
        if not mask.count(1):
            for n in range(self.SIZE):
                mask[index*self.SIZE + n] = 1

        return mask

    def get_mask_shcedule_route(self):
        state_graph = [[0 for column in range(self.SIZE)]
                      for row in range(self.SIZE)]
        for link in self.algo.topo.links:
            if link.isEntangled() and not link.taken:
                n1 = link.n1.id
                n2 = link.n2.id
                state_graph[n1][n2] = 1
                state_graph[n2][n1] = 1
        

        mask = [None for _ in range(self.algo.topo.numOfRequestPerRound * self.SIZE)]
        for reqState in self.algo.requestState:

            src,dst,current_node , path , index , done = reqState
            # print('in get mask +++==== ' , src.id,dst.id,current_node.id , path , index , done)

            if done:
                continue
            neighbors = [i for i, x in enumerate(state_graph[current_node.id]) if x == 1]
            for n in neighbors:
                if n not in path and n != current_node.id:
                    mask[index*self.SIZE + n] = 1
        # print('get mask ' , mask)
        if not mask.count(1):
            mask = self.get_mask__request_shcedule_route()
        return mask
    def get_mask__request_shcedule_route(self):
        mask = [None for _ in range(self.algo.topo.numOfRequestPerRound * self.SIZE)]
        for reqState in self.algo.requestState:

            src,dst,current_node , path , index , done = reqState
            # print('in get req mask +++==== ' , src.id,dst.id,current_node.id , path , index , done)
            if done:
                continue
            for n in range(self.SIZE):
                    mask[index*self.SIZE + n] = 1
        return mask
    
    def neighbor_qs_schedule_route(self , qs , mask = None):

            
        ret = []
        if mask is None:
            t = time.time()
            mask = self.get_mask_shcedule_route()
            # print('get mask time @@@@@@@@@  '  ,time.time()-t)
            # if self.algo.timeSlot%100 == 0:
            #     print(qs)
            #     print('average q val at: ' , self.algo.timeSlot , sum(qs) / len(qs))
            #     print(mask)
        min_ = -sys. maxsize - 1
        for i ,m in enumerate(mask):
            # print('in neighbor qs m:', m  , i , qs[i])
            if m is not None:
                ret.append(qs[i])
            else:
                ret.append(min_)
        # if self.algo.timeSlot%100 == 0:
        #     print(ret)
        return ret
    def rand_neighbor_schedule_route(self , mask = None):
        ret = []
        if mask is None:
            mask = self.get_mask_shcedule_route()
        for i ,m in enumerate(mask):
            if m is not None:
                ret.append(i)
        # print(len(ret) , mask)
        if not len(ret):
            # print('len ret is 0 ')
            mask = self.get_mask__request_shcedule_route()
            # print(mask)
            for i ,m in enumerate(mask):
                if m is not None:
                    ret.append(i)

        return random.choice(ret)







    def neighbor_qs(self , current_node_id , current_state , path , qs , neighbor_list = []):
        if current_node_id == self.SIZE:
            return qs
        r = random.random()
        # if  r < 0.01:
        #     print('==================')
        #     print('==================')
        #     print(r)
        #     print('==================')
        #     print('==================')
            
            # print(qs)
        for q in qs:
            if math.isnan(q):
                print('naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaan')
                quit()
        # print('==================')
        # print('==================')
        # print('==================')
        # print('==================')

        # neighbor_list = current_state[current_node_id]
        if not len(neighbor_list):
            neighbor_list = self.get_neighbors(current_node_id)
            # neighbor_list = [0 for _ in range(len(qs))]
            # current_node = self.algo.topo.nodes[current_node_id]
            # neighbors = [link.theOtherEndOf(current_node).id for link in current_node.links if link.isEntangled() and not link.taken]
            # for n in neighbors:
            #     neighbor_list[n] = 1
        # path = current_state[2*self.SIZE + 3]
        state_path = [0 for i in range(self.SIZE)]
        for i in range(len(path)):
            state_path[path[i]] = i + 1

        ret = []
        min_q = min(qs) 

        for i in range(len(neighbor_list)):
            n = neighbor_list[i]
            if n > 0 and state_path[i] == 0:
                ret.append(qs[i])
            else:
                ret.append(min_q)
        
        ret[current_node_id] = min_q

        if self.skip:
            ret.append(qs[-1])

        # print(ret)
        
        return ret
        # return qs
    def next_node_from_shortest(self , current_node_id , current_state , path , qs):
        r = random.random()


        # neighbor_list = current_state[current_node_id]
        neighbor_list = [0 for _ in range(self.SIZE)]
        current_node = self.algo.topo.nodes[current_node_id]
        neighbors = [link.theOtherEndOf(current_node).id for link in current_node.links if link.isEntangled() and not link.taken]
        for n in neighbors:
            neighbor_list[n] = 1
        dist = current_state[self.SIZE + 2]
        neighbor_dist = {}
        for i in range(len(neighbor_list)):
            if neighbor_list[i] > 0:
                neighbor_dist[i] = dist[i]
        # neighbor_dist = dict(sorted(neighbor_dist.items(), key=lambda item: item[1]))
        if not len(neighbor_dist):
            # print('= = r a n d = =')
            return np.random.randint(0, self.SIZE)
        ret  = min(neighbor_dist, key=neighbor_dist.get)
        
        return ret

    
    def rand_neighbor(self , current_node_id , current_state, path):
        # neighbor_list = current_state[current_node_id]
        neighbor_list = [0 for _ in range(self.SIZE)]
        current_node = self.algo.topo.nodes[current_node_id]
        neighbors = [link.theOtherEndOf(current_node).id for link in current_node.links if link.isEntangled() and not link.taken]
        for n in neighbors:
            neighbor_list[n] = 1
        # path = current_state[2*self.SIZE + 3]
        state_path = [0 for i in range(self.SIZE)]
        for i in range(len(path)):
            state_path[path[i]] = i + 1

        ret = []

        for i in range(len(neighbor_list)):
            n = neighbor_list[i]
            if n > 0 and state_path[i] == 0 :
                ret.append(i)
        if not len(ret):
            if self.skip:
                return np.random.randint(0, self.SIZE+1)
            return np.random.randint(0, self.SIZE)
        if len(path) == 1 and self.skip:
            ret.append(self.SIZE)
        return random.choice(ret)
        # return np.random.randint(0, self.SIZE)
    
    def max_future_q(self , qs, mask):
        # current_node_id = np.where(current_state[2*self.SIZE + 1] == 1)[0][0]
        # current_node_id = np.where(current_state[self.SIZE + 1] >= 1)[0][0]

        return np.max(self.neighbor_qs_schedule_route( qs , mask))
    # def max_future_q(self , qs):
    #     # current_node_id = np.where(current_state[2*self.SIZE + 1] == 1)[0][0]
    #     current_node_id = np.where(current_state[self.SIZE + 1] >= 1)[0][0]

    #     return np.max(self.neighbor_qs(current_node_id , current_state ,[], qs))
    def max_future_q_schedule(self,reqmask , qs):
        # print('in max_future_q_schedule ' , reqmask)
        ret = []
        for i in range(len(reqmask)):
            if not reqmask[i]:
                ret.append(qs[i])
        if not len(ret):
            return 0
        return max(ret)
    def rand_request(self , requests):
        ret = []
        for request in requests:
            if not request[2]:
                ret.append(request[4])
        # print(len(requests))
        return random.choice(ret)
    def get_next_request_id(self , requests , qs):
        print('average q val at: ' , self.algo.timeSlot , sum(qs) / len(qs))
        ret = {}
        for request in requests:
            if not request[2]:
                ret[request[4]] = qs[request[4]]
        # print(ret)
        return max(ret, key=ret.get)
        






