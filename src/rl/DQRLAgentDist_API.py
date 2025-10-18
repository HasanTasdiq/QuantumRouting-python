from itertools import islice
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
import gc
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras import Input

from collections import deque
import time
import random
import os
import multiprocessing
import math
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True 
import copy
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow as tf
import pickle
import glob
import dill
import sys
from objsize import get_deep_size


from keras.layers import Embedding, Flatten, Attention, Dense, MultiHeadAttention, LayerNormalization
from dist_agent_helper import executor, schedule_routing_state_dist , process_actions , replay_memory, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, START_EPSILON_DECAYING, END_EPSILON_DECAYING






NUM_EPISODES = 2500
LEARNING_RATE = .8
lr = .0001
clip_value = .1



GAMMA = 0.9
# GAMMA = 5
ALPHA = .9
BETA = -.1
DELTA = 0

ENTANGLEMENT_LIFETIME = 10
# Exploration settings

EPSILON_ = 1  # not a constant, qoing to be decayed



EPSILON_DECAY_VALUE = EPSILON_/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCOUNT = 0.5

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
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
model_lock = None
# executor = ThreadPoolExecutor(max_workers=8)



class DQRLAgentDist:
    def __init__(self , pid = 0):
        self.pid = pid
        # Check if a GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs are available: {gpus}")
        else:
            print("No GPUs detected. Running on CPU.")




    def initiate(self):
        # algo = self.algo
        print('++++++++++initiating DQRL agent for distributed routing++++++++++')

        # self.env = RoutingEnv(algo)
        self.SIZE = 100


        self.OBSERVATION_SPACE_VALUES = (128 + self.SIZE + 2 * (self.SIZE ** 2),1,)

        self.model_name = 'DQRL_dist_' + str(self.SIZE)
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())



        # An array with last n steps for training
        # self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.priorities = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.last_action_table = []
        self.reqState_qs = {}
        self.embedding_layer = Embedding(input_dim=20, output_dim=1)
        self.attention_layer = Attention()

        self.dense_proj = Dense(64, activation='relu')
        self.mha = MultiHeadAttention(num_heads=4, key_dim=16)
        self.ln = LayerNormalization()
    def print_weight(self , model):
        for r in model.get_weights():
            print(r)
    def create_model(self):
        # try:
        #     model = load_model(self.model_name)
        #     print('=====================================================model loaded from ',self.model_name,' =====================================')
        #     print(model.weights)
            
        #     # time.sleep(10)
        #     return model
        # except:
        #     print('=====================no model found========================')
        #     # time.sleep(10)

        
        model = Sequential()

        numAction = self.SIZE 
        numInput = self.OBSERVATION_SPACE_VALUES[0]*self.OBSERVATION_SPACE_VALUES[1]
        layer1 = int((math.sqrt(numInput) +2*numAction) //3)
        layer2 = int((math.sqrt(numInput) +numAction) //4)
        layer3 = int((math.sqrt(numInput) +2*numAction) //5)

        # layer1 = 300
        # layer2 = 200
        # layer3 = 100

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

        model.add(Dense(layer1, activation='relu'))
        model.add(Dense(layer3 , activation='relu'))



        model.add(Dense(self.SIZE, activation='linear')) 

        model.compile(loss="mse", optimizer=Adam(learning_rate = lr , clipvalue=clip_value), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition , priority):
        global replay_memory
        if type(transition) is list:
            replay_memory.extend(transition)
        else:
            replay_memory.append(transition)
        # self.priorities.append(priority)
    




    # Trains main network every step during episode
    def get_last_n(self , d, n):
        """Return last n elements of a deque efficiently"""
        return list(islice(d, len(d)-n, len(d)))
    
    def train(self, terminal_state):
        global replay_memory
        t1 = time.time()
        # global model_lock
        # if model_lock is None:
        #     model_lock = multiprocessing.Lock()

        # Start training only if certain number of samples is already saved
        print('----------len(self.replay_memory)----------------', len(replay_memory))
        print('----------size(self.replay memory)----------------', get_deep_size(replay_memory)/1024/1024 , 'MB')

        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        

        # Get a minibatch of random samples from memory replay table
        # priorities = np.array(self.priorities)
        # probabilities = priorities / priorities.sum()

        # indices = np.random.choice(len(self.replay_memory), MINIBATCH_SIZE, p=probabilities)
        # minibatch = [self.replay_memory[i] for i in indices]
        last_half = self.get_last_n(replay_memory, MINIBATCH_SIZE // 2)

        minibatch = random.sample(replay_memory, MINIBATCH_SIZE//2)
        minibatch.extend(last_half)
        batch_size = MINIBATCH_SIZE
        print('=============sample ===========' , time.time() - t1)

        t11 = time.time()


        


        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        # print(current_states)
        # with model_lock:
        current_qs_list = self.model.predict(current_states , verbose=0, batch_size=batch_size)
        print('=============current_qs_list predict ===========' , time.time() - t11)

        t2 = time.time()

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states , verbose=0, batch_size=batch_size)

        print('=============future_qs_list predict===========' , time.time() - t2)

        t3 = time.time()
        X = []
        y = []
        # print(len(minibatch))

        # Now we need to enumerate our batches
        for index, ( current_state, action, reward, new_current_state,mask ,  done) in enumerate(minibatch):
           
            if not done:
                max_future_q = self.max_future_q_dist( future_qs_list[index], mask)
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
        # with model_lock:
        hist = self.model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0, shuffle=True,)
        print('============= total train done===========' , time.time() - t1)
        print('=============only train done===========' , time.time() - t4)
        # Update target network counter every episode
        # if terminal_state:
        self.target_update_counter += 1
        print('self.target_update_counter', self.target_update_counter)

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            print('------------------self.model.get_weights()-------------------')

            # with model_lock:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def max_future_q_dist(self , qs, mask):


        return np.max(self.neighbor_qs_schedule_route( qs , mask))
    def neighbor_qs_schedule_route(self, qs, mask=None):
        if mask is None:
            try:
                mask = self.get_mask_shcedule_route()
            except:
                print('====================no mask found in neighbor_qs_schedule_route===============')
                mask = [1 for _ in range(len(qs))]
        # Convert mask to a NumPy array for efficient operations
        mask = np.array(mask)
        
        # Set qs values to a very low value where mask is None
        min_val = -sys.maxsize - 1
        masked_qs = np.where(mask == 1, qs, min_val)
    
        return masked_qs.tolist()
    def get_mask_shcedule_route(self,ent_matrix=None, req_matrix=None):

        
        state_graph = ent_matrix
        mask = [None for _ in range(len(req_matrix)//2 * self.SIZE)]
        for reqState in req_matrix:

            src,dst,current_node_id ,path , index , done = reqState
            # print('in get mask +++==== ' , src.id,dst.id,current_node.id , path , index , done)

            if done:
                continue
            neighbors = [i for i, x in enumerate(state_graph[current_node_id]) if x >= 1]
            for n in neighbors:
                if not path[n] and  n != current_node_id:
                    mask[index*self.SIZE + n] = 1
        # print('get mask ' , mask)
        if not mask.count(1):
            mask = self.get_mask__request_shcedule_route(req_matrix)
        return mask
    def get_mask__request_shcedule_route(self, req_matrix):
        mask = [None for _ in range(len(req_matrix)//2 * self.SIZE)]
        for reqState in req_matrix:

            src,dst,current_node , path , index , done = reqState
            # print('in get req mask +++==== ' , src.id,dst.id,current_node.id , path , index , done)
            if done:
                continue
            for n in range(self.SIZE):
                    mask[index*self.SIZE + n] = 1
        return mask
    def get_qs_batch(self, states):

        # print('---------- len sts --------------' , len(sts))
        
        return self.model.predict(np.array(states), verbose=0, batch_size=len(states))
    
    def get_qs(self,state):
        t = time.time()
        # ret = self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0,use_multiprocessing=True)[0]
        ret = self.model.predict_on_batch(np.array(state).reshape(-1, *state.shape))[0]

        # print('predict time&&&&&&&&  ' , time.time()-t)
        return ret



  
    
    def get_mask_one_req_schedule_route(self , reqState , ent_matrix=None, req_matrix=None):
        mask = [None for _ in range(self.SIZE)]
        

        src,dst,current_node_id , path , index , done = reqState
        state_graph = ent_matrix

        neighbors = [i for i, x in enumerate(state_graph[current_node_id]) if x >= 1]
        # print('in get mask +++==== ' , src.id,dst.id,current_node.id , neighbors)
        for n in neighbors:
            if not path[n] and n != current_node_id:
                mask[ n] = 1

        if not mask.count(1): #make a random mask
            mask = [1 for _ in range(self.SIZE)]

        return np.array(mask)
    def get_epsilon_linear(self , timeSlot, eps_start=EPSILON_):
        if timeSlot < START_EPSILON_DECAYING:
            return eps_start
        if timeSlot >= END_EPSILON_DECAYING:
            return 0
        epsilon = eps_start - eps_start  * (timeSlot / END_EPSILON_DECAYING)
        return max(0, epsilon)
    
    def learn_and_predict_next_req_node_single(self, reqIndex, ent_matrix , req_matrix,dist_matrix,timeSlot):
        if timeSlot > 0 and timeSlot % 100 == 0:
            try:
                self.model = load_model(self.model_name)
                print('model loaded from ' , self.model_name , ' at timeSlot ' , timeSlot)
            except:
                print('no model found to load!!!!!!!!!!!!!!!')    
        print('learn_and_predict_next_req_node_single called ' )
        req = req_matrix[reqIndex][:6]
        req[3] = req_matrix[reqIndex + len(req_matrix)//2]

        if req[5]:
            return None  # Request already checked/completed
        print('going to get current state')
        current_state = schedule_routing_state_dist(req , ent_matrix , req_matrix, dist_matrix)
        print('current_state going to get qs')
        qs = self.get_qs(current_state)
        print('going to get mask')
        mask = self.get_mask_one_req_schedule_route(req, ent_matrix , req_matrix)
        print('got mask')
        valid_actions = np.where(mask == 1)[0]
        valid_q_values = qs[valid_actions]

        # Sort valid actions based on Q values in descending order
        sorted_valid_actions = sorted(zip(valid_actions, valid_q_values), key=lambda x: x[1], reverse=True)
        sorted_valid_actions = [action for action, q in sorted_valid_actions]

        random_val = np.random.random()
        epsilon = self.get_epsilon_linear(timeSlot)
        if random_val > epsilon:
            print('======using greedy action from model ' , timeSlot , epsilon)
            action = np.argmax(np.where(mask == 1, qs, -np.inf))
        else:
            action = np.random.choice(valid_actions)
            random.shuffle(sorted_valid_actions)

        q = qs[action]
        # self.env.algo.action_count[action] += 1

        return [current_state.tolist() , int(action)]
   
    
    def decode_schdeule_route_action(self, action):
        request_index = math.floor(action / self.SIZE)
        next_node_id = action % self.SIZE
        return request_index , next_node_id
   


    def update_reward(self, numsuccessReq  , timeSlot , actionIds = None):
        global EPSILON_

        t1 = time.time()
        R = []


        reward = 0
        success = 0
        # pathlen = len(self.last_action_table[request])
        pathlen = 1
        req = []
        total_reward = 0
        trans = []
        print('++++++++++++++++++++++++before process action ', timeSlot )
        last_action_table = process_actions(actionIds)
        print('++++++++++++++++++++++++after process action ' , len(last_action_table), timeSlot , time.time()-t1 , 'seconds' )
        if True:
            for i in range(len(last_action_table)-1 , -1 , -1):
                t2 = time.time()
                (request , action , ts ,current_node_id, current_state , next_state ,mask ,  done,reward) = last_action_table[i]
                
                # req_id , next_node_id = self.decode_schdeule_route_action(action)
                # req.append(request)
                # print('before find reward time ')
                # reward = self.find_reward_routing(request  , timeSlot ,current_node_id , next_node_id)
                # print('after find reward time ' )
                # reward = self.env.find_reward_routing(request  , timeSlot ,current_node_id , action)
                # print((request[0].id , request[1].id) , reward)


                # if len(R):
                #     f = 0


                #     reward = reward * ALPHA + GAMMA * R[-1]

                #     reward /= pathlen
                #     # print((request[0].id , request[1].id) , reward)

                #     # R.append(reward)
                        
                # else:
                #     # reward = reward*ALPHA + numsuccessReq * GAMMA + avgFidelity* DELTA
                #     reward = reward*ALPHA + numsuccessReq * GAMMA 
                #     # reward = numsuccessReq
                #     reward /= pathlen
                #     R.append(reward)
                # reward = reward*ALPHA + numsuccessReq * GAMMA 
                reward = numsuccessReq

                # reward /=10
                total_reward += reward
                # print('get reward time ' , time.time() -t2)
                t3 = time.time()
                transition = ( current_state, action, reward, next_state,mask,  done)
                trans.append(transition)


                # print('update  replay memory time ' , time.time() -t3)
        t4 = time.time()
        print('before update replay memory time ' , time.time()-t4)
        self.update_replay_memory(trans, numsuccessReq)

        print('time for update memory ' , time.time()-t4)
        # self.env.algo.topo.reward_routing = {}
        t5 = time.time()


        ############################################
        # if timeSlot % 3 == 0:
        self.train(False )
        print('time train ' , time.time()-t5)

        # print('===---------size of model memory----------------===-' , get_deep_size(self.model)/1024/1024 , 'MB')
        # print('===---------size of target model memory----------------===-' , get_deep_size(self.target_model)/1024/1024 , 'MB')
        # print('==----------size(self.last_action_table memory)---------------==-', get_deep_size(self.last_action_table)/1024/1024 , 'MB')

        last_action_table = []
        gc.collect()
        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE
        # print(R)
        # print([(r[0].id,r[1].id) for r in req])

        # print('update_reward done in \n')
        # print('update_reward done in \n')
        # print('update_reward done in \n')
        # print('update_reward done in \n')
        # print('update_reward done in \n')
        # print('update_reward done in ' , time.time() - t1 , 'seconds\n')
        if timeSlot % 100 == 0:
            self.save_model()
            print('model saved at time slot ' , timeSlot)
        print('======!=======!==== total update reward done in ' , time.time() - t1 , 'seconds\n')
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
        # global model_lock
        # if model_lock is None:
        #     model_lock = multiprocessing.Lock()

        # with model_lock:
        
        self.model.save((self.model_name))
        # print(self.model.weights)
        # del self.model

if __name__ == '__main__':
    agent = DQRLAgentDist()
    agent.initiate()
    
    
