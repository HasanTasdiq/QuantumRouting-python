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

from keras.layers import Embedding, Flatten, Attention, Dense, MultiHeadAttention, LayerNormalization




# Check if a GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUs are available: {gpus}")
else:
    print("No GPUs detected. Running on CPU.")


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

# run 25k
START_EPSILON_DECAYING = 10000
END_EPSILON_DECAYING = 20000
REPLAY_MEMORY_SIZE = 20000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 1500  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)

# for 5k local
# START_EPSILON_DECAYING = 2000
# END_EPSILON_DECAYING = 4000
# REPLAY_MEMORY_SIZE = 12000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 1000  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 70  # Terminal states (end of episodes)

# for 3k local
# START_EPSILON_DECAYING = 1000
# END_EPSILON_DECAYING = 2500
# REPLAY_MEMORY_SIZE = 12000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 1000  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 70  # Terminal states (end of episodes)

#for 10k local
# START_EPSILON_DECAYING = 5000
# END_EPSILON_DECAYING = 8000
# REPLAY_MEMORY_SIZE = 15000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 2024  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)


# for testing
# START_EPSILON_DECAYING = 10
# END_EPSILON_DECAYING = 20
# REPLAY_MEMORY_SIZE = 2000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)

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
table_lock = multiprocessing.Lock()

class DQRLAgentDist:
    def __init__(self , pid = 0):
        self.pid = pid




    def initiate(self):
        # algo = self.algo
        print('++++++++++initiating DQRL agent for distributed routing++++++++++')

        # self.env = RoutingEnv(algo)
        self.SIZE = 25


        self.OBSERVATION_SPACE_VALUES = (128 + self.SIZE + 2 * (self.SIZE ** 2),1,)

        self.model_name = 'DQRL_dist_' + str(self.SIZE)
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())



        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
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
        if type(transition) is list:
            self.replay_memory.extend(transition)
        else:
            self.replay_memory.append(transition)
        # self.priorities.append(priority)
    
    def save_replay_memory(self, timeSlot):
        if not os.path.isdir('replay_memory'):
            os.makedirs('replay_memory')
        with open('replay_memory/' + self.model_name +'_'+ str(timeSlot) + '.pkl', 'wb') as f:
            pickle.dump(self.replay_memory, f)
        self.replay_memory.clear()
        self.priorities.clear()
        print('Replay memory saved')

    def load_replay_memory(self):
        if not os.path.isdir('replay_memory'):
            os.makedirs('replay_memory')
        replay_memory_files = glob.glob(f'replay_memory/{self.model_name}_*.pkl')
        self.replay_memory.clear()
        for file in replay_memory_files:
            try:
                with open(file, 'rb') as f:
                    self.replay_memory.extend(pickle.load(f))
                print(f'Replay memory loaded from {file}')
            except FileNotFoundError:
                print(f'No replay memory file found: {file}')
        print(f'Total replay memory loaded: {len(self.replay_memory)}')



    # Trains main network every step during episode
    def train(self, terminal_state):
        t1 = time.time()

        # Start training only if certain number of samples is already saved
        print('----------len(self.replay_memory)----------------', len(self.replay_memory))

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        # priorities = np.array(self.priorities)
        # probabilities = priorities / priorities.sum()

        # indices = np.random.choice(len(self.replay_memory), MINIBATCH_SIZE, p=probabilities)
        # minibatch = [self.replay_memory[i] for i in indices]
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        batch_size = MINIBATCH_SIZE
        print('=============sample ===========' , time.time() - t1)

        t11 = time.time()


        


        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        # print(current_states)
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



    # === 2. Request embeddings ===
    def get_request_embeddings(self, req_matrix):
        print('get_request_embeddings called' , len(req_matrix)//2)
        features = []
        for req in req_matrix:
            vec = [0] * self.SIZE
            if not req[5]:  # if not completed
                vec[req[0]] = 1  # current node
                vec[req[1]] = 10  # destination
            features.append(vec)
        print('get_request_embeddings before return')
        return tf.convert_to_tensor(features, dtype=tf.float32)


    # === 3. Request-level attention ===
    def apply_request_attention(self, request_tensor):


        # Project to 64D
        proj = self.dense_proj(request_tensor)  # shape: [num_requests, 64]

        # Add batch dimension
        proj = tf.expand_dims(proj, axis=0)  # shape: [1, num_requests, 64]

        # Apply MHA
        attn_out = self.mha(query=proj, key=proj, value=proj)  # Correct usage

        # Residual + LayerNorm
        output = self.ln(proj + attn_out)  # shape: [1, num_requests, 64]

        return tf.squeeze(output, axis=0)  # shape: [num_requests, 64]

    # === 4. Neighbor embedding ===
    def get_neighbor_embeddings(self, state_graph, current_node_id):
        neighbors = state_graph[current_node_id]
        neighbor_feats = []
        for node_id, has_link in enumerate(neighbors):
            if has_link > 0:
                feat = [0] * self.SIZE
                feat[node_id] = 1  # one-hot neighbor
                vec = tf.convert_to_tensor(feat, dtype=tf.float32)
                vec = tf.expand_dims(vec, axis=0)  # (1, SIZE)
                vec = Dense(64, activation='relu')(vec)
                vec = tf.squeeze(vec, axis=0)      # (64,)
                neighbor_feats.append(vec)
        return neighbor_feats



    # === 5. Neighbor attention ===
    def apply_neighbor_attention(self, curr_emb, neighbor_embs):
        if not neighbor_embs:
            return np.zeros(64)
        stack = tf.stack(neighbor_embs)  # [num_neighbors, 64]
        query = tf.expand_dims(curr_emb, axis=0)  # [1, 64]
        scores = tf.matmul(query, stack, transpose_b=True) / tf.math.sqrt(64.0)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, stack)[0].numpy()
        return context


    # === 6. Main function ===
    def schedule_routing_state_dist(self, curr_req, ent_matrix=None, req_matrix=None, dist_matrix=None):


        # 1. Link matrices
        print('in schedule_routing_state_dist' )
        state_graph, state_dist = ent_matrix, dist_matrix
        print('state_graph found')

        state_graph_flat = np.array(state_graph).flatten()  # shape: [SIZE × SIZE]
        state_dist_flat = np.array(state_dist).flatten()    # shape: [SIZE × SIZE]
        print('state_graph_flat found')
        # 2. Request embeddings
        req_tensor = self.get_request_embeddings(req_matrix)
        print('req_tensor found')

        # 3. Apply self-attention
        attn_encoded = self.apply_request_attention(req_tensor).numpy()
        print('attn_encoded found')
        # 4. Locate current request
        curr_index = curr_req[4]
        print('curr_index found' , curr_index)

        curr_emb = attn_encoded[curr_index]

        # 5. Neighbor context
        neighbor_embs = self.get_neighbor_embeddings(state_graph, curr_req[2])
        print('neighbor_embs found' , len(neighbor_embs))
        context_vec = self.apply_neighbor_attention(curr_emb, neighbor_embs)
        print('context_vec found')

        # 6. Local info
        local = [0] * self.SIZE
        local[curr_req[2]] = 10
        local[curr_req[1]] = 10

        # 7. Final state vector
        ret = np.concatenate([
            curr_emb,         # attention-aware request embedding
            context_vec,      # neighbor context
            np.array(local),   # current and destination
            state_graph_flat,
            state_dist_flat
        ])

        # print(ret)
        # exit()

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

    
    def learn_and_predict_next_req_node_single(self, reqIndex, ent_matrix , req_matrix,dist_matrix):
        global EPSILON_
        req = req_matrix[reqIndex][:6]
        req[3] = req_matrix[reqIndex + len(req_matrix)//2]

        if req[5]:
            return None  # Request already checked/completed

        current_state = self.schedule_routing_state_dist(req , ent_matrix , req_matrix, dist_matrix)
        print('current_state going to get qs')
        qs = self.get_qs(current_state)
        mask = self.get_mask_one_req_schedule_route(req, ent_matrix , req_matrix)
        valid_actions = np.where(mask == 1)[0]
        valid_q_values = qs[valid_actions]

        # Sort valid actions based on Q values in descending order
        sorted_valid_actions = sorted(zip(valid_actions, valid_q_values), key=lambda x: x[1], reverse=True)
        sorted_valid_actions = [action for action, q in sorted_valid_actions]

        random_val = np.random.random()
        if random_val > EPSILON_:
            action = np.argmax(np.where(mask == 1, qs, -np.inf))
        else:
            action = np.random.choice(valid_actions)
            random.shuffle(sorted_valid_actions)

        q = qs[action]
        # self.env.algo.action_count[action] += 1

        return [current_state.tolist() , action]
   
    
    def decode_schdeule_route_action(self, action):
        request_index = math.floor(action / self.SIZE)
        next_node_id = action % self.SIZE
        return request_index , next_node_id
    def update_action(self , request_index ,current_node_id,  action  , current_state  , done , timeSlot,lreward,ent_matrix , req_matrix,dist_matrix):
        global EPSILON_
        request = req_matrix[request_index][:6]
        request[3] = req_matrix[request_index + len(req_matrix)//2]

        # print('doooooooooooooooooooone -------------- ' , done , (request[0].id , request[1].id) ,current_node_id , action)
        if not done:
            t = time.time()
            next_state = self.schedule_routing_state_dist(request, ent_matrix , req_matrix,dist_matrix)
            # print('update action get state time ' , time.time()-t)
        else:
            next_state = None

        if next_state is None:
            print('next state is none in update action!!!!!!!!!!!!!')
            next_state = current_state
        # done = False
        t = time.time()
        mask = self.get_mask_one_req_schedule_route(request,ent_matrix , req_matrix) #action is the next node id
        # print('update action get get_mask_shcedule_route time ' , time.time()-t)
        t = time.time()
        with table_lock:
            self.last_action_table.append((request , action , timeSlot ,current_node_id,  current_state , next_state ,mask ,  done, lreward))
        # print('update action  last_action_table.append( time ' , time.time()-t)


    
    def update_reward(self, numsuccessReq  , timeSlot):
        global EPSILON_

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
        with table_lock:
            for i in range(len(self.last_action_table)-1 , -1 , -1):
                t2 = time.time()
                (request , action , ts ,current_node_id, current_state , next_state ,mask ,  done,reward) = self.last_action_table[i]
                
                # req_id , next_node_id = self.decode_schdeule_route_action(action)
                # req.append(request)
                print('before find reward time ')
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
                print('get reward time ' , time.time() -t2)
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
        self.train(False )
        print('time train ' , time.time()-t5)



        self.last_action_table = []
        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE
        # print(R)
        # print([(r[0].id,r[1].id) for r in req])

        print('update_reward done in \n')
        print('update_reward done in \n')
        print('update_reward done in \n')
        print('update_reward done in \n')
        print('update_reward done in \n')
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
        # print(self.model.weights)
        del self.model

if __name__ == '__main__':
    agent = DQRLAgentDist()
    agent.initiate()
    
    
