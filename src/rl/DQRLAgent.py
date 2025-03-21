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
NUM_EPISODES = 2500
LEARNING_RATE = 0.001


GAMMA = 0.99

ENTANGLEMENT_LIFETIME = 10
# Exploration settings

EPSILON_ = 0.9  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 20000
EPSILON_DECAY_VALUE = EPSILON_/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 10000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20



#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
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
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE *2 + 3,self.env.SIZE,)  
        self.OBSERVATION_SPACE_VALUES = (self.env.SIZE + 3 + self.env.algo.topo.numOfRequestPerRound,self.env.SIZE,)  
        self.model_name = algo.name+'_'+ str(len(algo.topo.nodes)) +'_'+str(algo.topo.alpha) +'_'+str(algo.topo.q) +'_'+'DQRLAgent.keras'

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # print(self.model.get_weights())
        self.print_weight(self.model)

        # print(self.target_model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.last_action_table = {}
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



        model.add(Flatten(input_shape = self.OBSERVATION_SPACE_VALUES))  
        # model.add(Dense(self.env.SIZE * 20 , activation='relu'))
        model.add(Dense(self.env.SIZE * 10 , activation='relu'))
        # model.add(Dense(72 , activation='relu'))
        model.add(Dense(self.env.SIZE * 5 , activation='relu'))

        # model.add(Conv2D(32, 3, activation="relu"))
        # model.add(Flatten)

        model.add(Dense(self.env.SIZE, activation='linear')) 
        print(model.summary)
        # print('------------------self.model.get_weights()-------------------')
        # print(model.get_weights())

        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.compile(loss="mse", optimizer=Adam(learning_rate = 0.0002, clipvalue=0.01 ), metrics=['accuracy'])
        # model._make_predict_function()
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        print('----------len(self.replay_memory)----------------', len(self.replay_memory))
        # print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        
        # print([(m[1] , m[2]) for m in minibatch])

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states , verbose=0, batch_size=64)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states , verbose=0, batch_size=64)

        X = []
        y = []
        # print(len(minibatch))

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # print('---action-- ' , action)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                # max_future_q = np.max(future_qs_list[index])

                max_future_q = self.env.max_future_q(new_current_state , future_qs_list[index])
                
                # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index][action])
                # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # print(new_q)
            # print(current_state)

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # print('=============train start===========')
        t1 = time.time()
        # Fit on all samples as one batch, log only on terminal state
        hist = self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=True, )
        print('=============train done===========' , time.time() - t1)
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
            # print(qs)
            
            if np.random.random() > EPSILON_:
                next_node = np.argmax(self.env.neighbor_qs(current_node.id , current_state, path , self.get_qs(current_state)))

                q = qs[next_node]
                # print('--- get_qs-- ' , time.time() - t2 , 'seconds')

            else:
                # Get random action
                next_node = self.env.rand_neighbor(current_node.id , current_state,path)

                q = qs[next_node]

            reqState_action_q.append((reqState , next_node , q , current_state))
        
        # if np.random.random() > EPSILON_:
        #     link_action_q.sort(key=lambda x: x[2], reverse=True)

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
    
    def update_action(self , request ,  action  , current_state , path , done):
        global EPSILON_

        timeSlot = self.env.algo.timeSlot

        next_state = self.env.routing_state(request , action , path ,  timeSlot)

        if next_state is None:
            next_state = current_state
        if not request in self.last_action_table:
            self.last_action_table[request] = [(action , timeSlot , current_state , next_state , done)]
        else:
            self.last_action_table[request].append((action , timeSlot , current_state , next_state , done))

        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE
    
    def update_reward(self):
        print('update reward DQRA :::::::::::::::::::::::: ' , len(self.last_action_table) )
        t1 = time.time()
        for request in self.last_action_table:

            reward = 0
            for i in range(len(self.last_action_table[request])):

                (action , timeSlot , current_state , next_state , done) = self.last_action_table[request][i]
                # current_node_id = current_state[2*self.env.SIZE + 1].index(1)
                # current_node_id = np.where(current_state[2*self.env.SIZE + 1] == 1)[0][0]
                # print('update_reward for:: ', current_state[self.env.SIZE + 1])

                current_node_id = np.where(current_state[self.env.SIZE + 1] >= 1)[0][0]
                # print(current_state[2*self.env.SIZE + 1], current_node_id, str(current_node_id))
                reward = self.env.find_reward_routing(request  , timeSlot ,current_node_id , action)
                if not reward:
                    continue

                self.update_replay_memory((current_state, action, reward, next_state, done))
        self.env.algo.topo.reward_routing = {}
        self.train(False , self.env.algo.timeSlot)


        #need to fix!!!!!!!!!!!!!!!!!!!!!!!!
        # for request in self.last_action_table:
        #     # print(self.last_action_table[pair])
        #     self.last_action_table[request] = list(filter(lambda x: self.env.algo.timeSlot -  x[1] < ENTANGLEMENT_LIFETIME , self.last_action_table[pair]))
        self.last_action_table = {}
        print('update_reward done in ' , time.time() - t1 , 'seconds\n')
    
    def save_model(self):
        self.model.save((self.model_name))
