import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras import Input

from collections import deque
import time
import random
import os
from RoutingEnv import RoutingEnv      #for ubuntu
# from .RoutingEnv import RoutingEnv   #for mac
import multiprocessing
import math
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True 
from objsize import get_deep_size
import copy
NUM_EPISODES = 2500
LEARNING_RATE = .8
lr = .0001
clip_value = .1


GAMMA = 0.9
# GAMMA = 5
ALPHA = .9
BETA = -.1

ENTANGLEMENT_LIFETIME = 10
# Exploration settings

EPSILON_ = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 30000
END_EPSILON_DECAYING = 45000
EPSILON_DECAY_VALUE = EPSILON_/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCOUNT = 0.5
REPLAY_MEMORY_SIZE = 200000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 2024  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 50  # Terminal states (end of episodes)
FAILURE_REWARD = -2
# SKIP_REWAD = -2
SKIP_REWAD = -2
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20 



#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = False


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)

# Memory fraction, used mostly when training multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class DQRLAgent:
    def __init__(self ,algo , pid):
        print('++++++++++initiating DQRL agent for:' , algo.name)
        self.env = RoutingEnv(algo)
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE *2 +2,self.env.SIZE,)  
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE  + 3,self.env.SIZE,)  
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE + 2 + self.env.algo.topo.numOfRequestPerRound,self.env.SIZE,)  
        # self.OBSERVATION_SPACE_VALUES = (self.env.algo.topo.numOfRequestPerRound , 4*self.env.SIZE + self.env.SIZE*self.env.SIZE + self.env.SIZE,)  
        self.OBSERVATION_SPACE_VALUES = (self.env.algo.topo.numOfRequestPerRound , 4*self.env.SIZE + self.env.SIZE*self.env.SIZE ,)  

        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE + 3 + self.env.SIZE,self.env.SIZE,)  
        
        # self.OBSERVATION_SPACE_VALUES = (self.env.algo.topo.numOfRequestPerRound , 3*self.env.SIZE + self.env.SIZE*self.env.SIZE ,)  
       
        self.model_name = algo.name+'_'+ str(len(algo.topo.nodes)) +'_'+str(algo.topo.alpha) +'_'+str(algo.topo.q) +'_'+'DQRLAgent.keras'

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # print(self.model.get_weights())
        # self.print_weight(self.model)

        # print(self.target_model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.priorities = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.last_action_table = []
        self.reqState_qs = {}




    def print_weight(self , model):
        for r in model.get_weights():
            print(r)
    def create_model(self):
        # try:
        #     model = load_model(self.model_name)
        #     print('=====================================================model loaded from ',self.model_name,' =====================================')
        #     # time.sleep(10)
        #     return model
        # except:
        #     print('=====================no model found========================')
        #     # time.sleep(10)

        
        model = Sequential()

        # model.add(Conv1D(256, (3, ), input_shape=self.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=(2, )))
        # model.add(Dropout(0.2))

        # model.add(Conv1D(256, (3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=(2, )))
        # model.add(Dropout(0.2))

        # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(Dense(64))

        numAction = self.env.SIZE * self.env.algo.topo.numOfRequestPerRound
        numInput = self.OBSERVATION_SPACE_VALUES[0]*self.OBSERVATION_SPACE_VALUES[1]
        layer1 = (math.sqrt(numInput) +2*numAction) //3
        layer2 = (math.sqrt(numInput) +numAction) //4
        layer3 = (math.sqrt(numInput) +2*numAction) //5

        print('==============')
        print('==============')
        print('==============')
        print('==============')
        print(numInput)
        print(numAction)
        print(layer1)
        print(layer2)
        print(layer3)
        print('==============')
        print('==============')
        print('==============')
        print('==============')
        # exit()


        model.add(Flatten(input_shape = self.OBSERVATION_SPACE_VALUES))  
        # model.add(Dense(self.env.SIZE * 20 , activation='relu'))
        # model.add(Dense(self.env.SIZE * 10 , activation='relu'))
        # # model.add(Dense(72 , activation='relu'))
        # model.add(Dense(self.env.SIZE * 5 , activation='relu'))
        model.add(Dense(layer1, activation='relu'))
        model.add(Dense(layer2 , activation='relu'))
        model.add(Dense(layer3 , activation='relu'))

        # model.add(Conv2D(32, 3, activation="relu"))
        # model.add(Flatten)

        model.add(Dense(self.env.SIZE * self.env.algo.topo.numOfRequestPerRound, activation='linear')) 
        # model.add(Dense(self.env.SIZE , activation='linear')) 
        # print(model.summary)
        # print('------------------self.model.get_weights()-------------------')
        # print(model.get_weights())

        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        # model.compile(loss="mse", optimizer=Adam(learning_rate = lr, clipvalue=0.01 ), metrics=['accuracy'])
        model.compile(loss="mse", optimizer=Adam(learning_rate = lr , clipvalue=clip_value), metrics=['accuracy'])
        # model._make_predict_function()
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition , priority):
        if type(transition) is list:
            self.replay_memory.extend(transition)
        else:
            self.replay_memory.append(transition)
        self.priorities.append(priority)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        t1 = time.time()

        # Start training only if certain number of samples is already saved
        print('----------len(self.replay_memory)----------------', len(self.replay_memory))
        # print('----------size(self.replay_memory)----------------', get_deep_size(self.replay_memory)/1000000)
        print('get deep size time ' , time.time()-t1)

        # print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        # priorities = np.array(self.priorities)
        # probabilities = priorities / priorities.sum()

        # indices = np.random.choice(len(self.replay_memory), MINIBATCH_SIZE, p=probabilities)
        # minibatch = [self.replay_memory[i] for i in indices]
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        batch_size = 512
        print('=============sample ===========' , time.time() - t1)

        t11 = time.time()


        
        # print([(m[1] , m[2]) for m in minibatch])

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        # print(current_states)
        current_qs_list = self.model.predict(current_states , verbose=0, batch_size=batch_size,use_multiprocessing=True)
        print('=============current_qs_list predict ===========' , time.time() - t11)

        t2 = time.time()

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states , verbose=0, batch_size=batch_size,use_multiprocessing=True)

        print('=============future_qs_list predict===========' , time.time() - t2)

        t3 = time.time()
        X = []
        y = []
        # print(len(minibatch))

        # Now we need to enumerate our batches
        for index, ( current_state, action, reward, new_current_state,mask ,  done) in enumerate(minibatch):
            # print('---action-- ' , action)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # if not done:
            #     # max_future_q = np.max(future_qs_list[index])

            #     max_future_q = self.env.max_future_q(current_node_id , new_current_state , future_qs_list[index])
                
            #     # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index][action])
            #     # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index])
            #     new_q = reward + DISCOUNT * max_future_q
            # else:
            #     # new_q = reward

            #     max_future_q = self.env.max_future_q(current_node_id , new_current_state , future_qs_list[index])
                
            #     # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index][action])
            #     # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index])
            #     new_q = reward + DISCOUNT * max_future_q
            if not done:
                max_future_q = self.env.max_future_q( future_qs_list[index], mask)
                qval = current_qs_list[index][action]
                    
                # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index][action])
                # print('+++++++++++++++lr+++++++++++++ ' , reward , max_future_q, qval)
                new_q = (1-LEARNING_RATE)*qval + LEARNING_RATE *(reward + DISCOUNT * max_future_q)
                # new_q = reward + DISCOUNT * max_future_q
                # print(new_q)
                # print(current_state)
            else:
                qval = current_qs_list[index][action]

                new_q = (1-LEARNING_RATE)*qval + LEARNING_RATE *reward 
                # new_q = reward



            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            # print('------------------------------------------------------------------------ ' , action)

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        print('=============train prep done===========' , time.time() - t3)
        t4 = time.time()
        # print('=============train start===========')
        # Fit on all samples as one batch, log only on terminal state
        hist = self.model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0, shuffle=True, use_multiprocessing=True,)
        print('============= total train done===========' , time.time() - t1)
        print('=============only train done===========' , time.time() - t4)
        # Update target network counter every episode
        # if terminal_state:
        self.target_update_counter += 1
        print('self.target_update_counter', self.target_update_counter)

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            print('------------------self.model.get_weights()-------------------')
            # print(self.model.get_weights())
            # self.print_weight(self.model)
            # time.sleep(2)

            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)

    def get_qs_batch(self, states):
        
        sts = list()
        c = 0
        for reqState , state in states:
            # sts.append(np.array(state).reshape(-1, *state.shape))
            sts.append(state)
            # c+= 1
            # if c > 100:
            #     break
        # print(sts)
        
        # print('---------- len sts --------------' , len(sts))
        return self.model.predict(np.array(sts), verbose=0, batch_size=50)
    
    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]

    
    
    # def update_last_action_table(self , action , timeSlot , current_state , next_state):


    # def learn_and_predict(self):
    #     global EPSILON_
    #     t1 = time.time()
    #     edges = self.env.algo.topo.edges
    #     timeSlot = self.env.algo.timeSlot
    #     link_action_q = []


    #     # -----sequntial prediction----

       

    #     # for pair in state:
    #     for link in edges:
    #         self.get_link_qs_batch([link] , timeSlot)
    #         current_state , qs = self.link_qs[link]
    #         if np.random.random() > EPSILON_:
    #             # Get action from Q net
    #             # print('------self.get_qs(current_state)-----')
    #             # print(self.get_qs(current_state))
    #             t2 = time.time()

    #             action = np.argmax(qs)
    #             q = qs[action]
    #             # print('--- get_qs-- ' , time.time() - t2 , 'seconds')

    #         else:
    #             # Get random action
    #             action = np.random.randint(0, 2)
    #             q = qs[action]

    #         next_state  = self.env.assignQubitEdge(link , action , timeSlot)
    #         if next_state is None:
    #             next_state = current_state
            
    #         if not link in self.last_action_table:
    #             self.last_action_table[link] = [(action , timeSlot , current_state , next_state)]
    #         else:
    #             self.last_action_table[link].append((action , timeSlot , current_state , next_state))
        

    #     if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
    #         EPSILON_ -= EPSILON_DECAY_VALUE

    #     self.link_qs = {}
    #     print('**learn_and_predict multicore ent dqrl done in ' , time.time() - t1 , 'seconds')

    # def learn_and_predict(self):
    #     assignable = True
    #     while assignable:
    #         assignable = self.learn_and_predict2()
    #         if 'no_repeat' in self.env.algo.name:
    #             break
    def get_next_node_qs_batch(self , requestStates , timeSlot):
        # print('in get p q')
        if not len(requestStates):
            return
        states = []
        for reqState in requestStates:
            request = (reqState[0] , reqState[1])
            current_node = reqState[2]
            path = reqState[3]
            current_state = self.env.routing_state(request , current_node.id ,path ,  timeSlot )

            # print(current_state)
            states.append((reqState , current_state))

        # print('getting qs')
        # print(states)
        qs = self.get_qs_batch(states)
        # print('getting qs done! '  , len(qs))
        for i in range(len(states)):
            reqState , current_state = states[i]
            self.reqState_qs[reqState] = (current_state , qs[i]) #wip
    def learn_and_predict_next_node_batch(self , requestStates):
        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        reqState_action_q = []

        self.get_next_node_qs_batch(requestStates , timeSlot)

        for reqState in self.reqState_qs:

            current_state , qs = self.reqState_qs[reqState]
            current_node = reqState[2]
            path = reqState[3]
            index = reqState[4]
            # print(qs)
            
            if np.random.random() > EPSILON_:
                # next_node = np.argmax(self.env.neighbor_qs(current_node.id , current_state, path , self.get_qs(current_state)))
                next_node = np.argmax(self.env.neighbor_qs(current_node.id , current_state, path , qs))

                q = qs[next_node]
                # print('--- get_qs-- ' , time.time() - t2 , 'seconds')

            else:
                # Get random action
                if np.random.random() > 0 :
                    next_node = self.env.rand_neighbor(current_node.id , current_state,path)
                else:
                    next_node = self.env.next_node_from_shortest(current_node.id , current_state, path , qs)


                q = qs[next_node]
            # next_node = self.env.next_node_from_shortest(current_node.id , current_state, path , qs)
            # q = qs[next_node]


            reqState_action_q.append((reqState , next_node , q , current_state))
        
        # if np.random.random() > EPSILON_:
        #     link_action_q.sort(key=lambda x: x[2], reverse=True)
            
        if np.random.random() > EPSILON_:
            reqState_action_q.sort(key=lambda x: x[2], reverse=True)
        self.reqState_qs = {}

        return reqState_action_q
    
    def learn_and_predict_next_node_batch_shortest(self , requestStates):
        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        reqState_action_q = []

        self.get_next_node_qs_batch(requestStates , timeSlot)

        for reqState in self.reqState_qs:

            current_state , qs = self.reqState_qs[reqState]
            current_node = reqState[2]
            path = reqState[3]
            index = reqState[4]
            # print(qs)
            
            next_node = self.env.next_node_from_shortest(current_node.id , current_state, path , qs)
            # next_node = np.argmin(current_state[self.env.SIZE + 2])
            # print('========= ' , next_node)
            # print(current_state)
            q = qs[next_node]
                # print('--- get_qs-- ' , time.time() - t2 , 'seconds')


            reqState_action_q.append((reqState , next_node , q , current_state))
        
        # if np.random.random() > EPSILON_:
        #     link_action_q.sort(key=lambda x: x[2], reverse=True)
            
        if np.random.random() > EPSILON_:
            reqState_action_q.sort(key=lambda x: x[2], reverse=True)
        self.reqState_qs = {}

        return reqState_action_q


    def learn_and_predict_next_node(self , request , current_node , path):
        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        current_state = self.env.routing_state(request , current_node.id ,path ,  timeSlot )
        if np.random.random() > EPSILON_:

            t2 = time.time()

            # next_node = np.argmax(self.get_qs(current_state))
            next_node = np.argmax(self.env.neighbor_qs(current_node.id , current_state, path , self.get_qs(current_state)))
            
        else:  
            # next_node = np.random.randint(0, self.env.SIZE)
            next_node = self.env.rand_neighbor(current_node.id , current_state,path)
        
        return current_state, next_node
    
    def learn_and_predict_next_req_node(self):
        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        current_state = self.env.schedule_routing_state()

        if np.random.random() > EPSILON_:

            t2 = time.time()

            # next_node = np.argmax(self.get_qs(current_state))
            action = np.argmax(self.env.neighbor_qs_schedule_route(self.get_qs(current_state)))
            
        else:  
            # next_node = np.random.randint(0, self.env.SIZE)
            action = self.env.rand_neighbor_schedule_route()
        self.env.algo.action_count[action] += 1
        # print('aaaaccccttttiiioooonnn ' , action)
        # print(request_index , next_node_id)
        
        return current_state, action
    def learn_and_predict_next_req_node_all(self):
        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        current_state = self.env.schedule_routing_state()
        qs = self.get_qs(current_state)
        if self.env.algo.timeSlot%100 == 0:
            print(qs)
            print('average q val at: ' , self.env.algo.timeSlot , sum(qs) / len(qs))
        ret = []
        for req in self.env.algo.requestState:
            mask = self.env.get_mask_one_req_schedule_route(req)
            # print(req[4] , req[5] , mask)
            if np.random.random() > EPSILON_:

                t2 = time.time()

                # next_node = np.argmax(self.get_qs(current_state))
                action = np.argmax(self.env.neighbor_qs_schedule_route(qs , mask))
                
            else:  
                # next_node = np.random.randint(0, self.env.SIZE)
                action = self.env.rand_neighbor_schedule_route(mask)
            if req[5]:
                action = -1
            q = qs[action]
            ret.append([current_state , action , q  , mask])
            self.env.algo.action_count[action] += 1
        # print('aaaaccccttttiiioooonnn ' , action)
        # print(request_index , next_node_id)
        # print([a[1] for a in ret])
        return ret
    
    def decode_schdeule_route_action(self, action):
        request_index = math.floor(action / self.env.SIZE)
        next_node_id = action % self.env.SIZE
        return request_index , next_node_id
    def update_action(self , request ,current_node_id,  action  , current_state  , done):
        global EPSILON_

        timeSlot = self.env.algo.timeSlot
        # print('doooooooooooooooooooone -------------- ' , done , (request[0].id , request[1].id) ,current_node_id , action)
        if not done:
            t = time.time()
            next_state = self.env.schedule_routing_state()
            # print('update action get state time ' , time.time()-t)
        else:
            next_state = None

        if next_state is None:
            next_state = current_state
        # done = False
        t = time.time()
        mask = self.env.get_mask_shcedule_route() #action is the next node id
        # print('update action get get_mask_shcedule_route time ' , time.time()-t)
        t = time.time()
        self.last_action_table.append((request , action , timeSlot ,current_node_id,  current_state , next_state ,mask ,  done))
        # print('update action  last_action_table.append( time ' , time.time()-t)


    
    def update_reward(self, numsuccessReq):
        global EPSILON_

        timeSlot = self.env.algo.timeSlot
        print('update reward DQRA :::::::::::::::::::::::: ' , len(self.last_action_table) )
        t1 = time.time()
        R = []


        reward = 0
        success = 0
        # pathlen = len(self.last_action_table[request])
        pathlen = 1
        req = []
        total_reward = 0
        trans = []
        for i in range(len(self.last_action_table)-1 , -1 , -1):
            t2 = time.time()
            (request , action , timeSlot ,current_node_id, current_state , next_state ,mask ,  done) = self.last_action_table[i]
            req_id , next_node_id = self.decode_schdeule_route_action(action)
            req.append(request)
            reward = self.env.find_reward_routing(request  , timeSlot ,current_node_id , next_node_id)
            # reward = self.env.find_reward_routing(request  , timeSlot ,current_node_id , action)
            # print((request[0].id , request[1].id) , reward)


            if len(R):
                f = 0


                reward = reward * ALPHA + GAMMA * R[-1]

                reward /= pathlen
                # print((request[0].id , request[1].id) , reward)

                # R.append(reward)
                    
            else:
                reward = reward*ALPHA + numsuccessReq * GAMMA
                # reward = numsuccessReq
                reward /= pathlen
                R.append(reward)
            reward /=10
            total_reward += reward
            # print('get reward time ' , time.time() -t2)
            t3 = time.time()
            transition = ( current_state, action, reward, next_state,mask,  done)
            trans.append(transition)


            # print('update  replay memory time ' , time.time() -t3)
        t4 = time.time()
        self.update_replay_memory(trans, numsuccessReq)
        # if timeSlot == 1:
        #     for i in range(80000):
        #         self.update_replay_memory(copy.deepcopy( (current_state, action, i, next_state,mask,  done)), numsuccessReq)
            
            
            # print(R)
        print('time for update memory ' , time.time()-t4)
        self.env.algo.topo.reward_routing = {}
        t5 = time.time()

        self.train(False , self.env.algo.timeSlot)
        print('time train ' , time.time()-t5)



        self.last_action_table = []
        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE
        # print(R)
        # print([(r[0].id,r[1].id) for r in req])
        print('update_reward done in ' , time.time() - t1 , 'seconds\n')

        return total_reward

    def getOrderedRequests(self , paths):
        req_q = {req: 0 for req in paths}
        for req in paths:
            p = paths[req]
            for obj in p:
                req_q[req] += obj['q_val']
            req_q[req] = req_q[req] / (len(p) - 1)
        

        req_q = dict(sorted(req_q.items(), key=lambda item: item[1], reverse=True))

        T = list(req_q.keys())
        if np.random.random() > EPSILON_:
            random.shuffle(T)
        return T
    def save_model(self):
        self.model.save((self.model_name))
