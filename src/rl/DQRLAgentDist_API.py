from itertools import islice
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
import gc
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras import Input,Model, layers

from collections import defaultdict, deque
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
from dist_agent_helper import (
    schedule_routing_state_dist, process_actions, replay_memory,
    REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE,
    UPDATE_TARGET_EVERY, START_EPSILON_DECAYING, END_EPSILON_DECAYING,
    load_model_from_redis, save_model_to_redis, save_worker_model_to_redis,
    save_model_to_disk,
    TRAINING_MODE, INFERENCE_MODE, MAX_REQUESTS_SMOKE, MAX_REQUESTS_PAPER,
    SIZE, get_request_embeddings, apply_request_attention,
    get_neighbor_embeddings, apply_neighbor_attention,
)
from GNN import QRoutingGATFlat





NUM_EPISODES = 2500
LEARNING_RATE = .8
lr = .0001
clip_value = .1

GAMMA = 0.9
ALPHA = .9
BETA = -.1
DELTA = 0

# Paper Eq. 9 reward weights: R = r·λ + N_success·μ + F_avg·ν
REWARD_LAMBDA = 0.3
REWARD_MU = 1.0
REWARD_NU = 0.5

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
        self.loaded_ts = set()


        # Mode-aware cap: smoke=15 (covers req 5,10), paper=200
        self.MAX_REQUESTS = MAX_REQUESTS_SMOKE if TRAINING_MODE == "smoke" else MAX_REQUESTS_PAPER
        # --- QMIX Mixers ---
        # Global state = flattened ent_matrix (SIZE×SIZE) + one-hot node vec (SIZE)
        self.GLOBAL_STATE_DIM = self.SIZE * self.SIZE + self.SIZE   # e.g. 10100 for SIZE=100
        mixer_state_dim = self.GLOBAL_STATE_DIM

        self.mixer = QMixer(self.MAX_REQUESTS, mixer_state_dim)
        self.target_mixer = QMixer(self.MAX_REQUESTS, mixer_state_dim)

        # Pre-build mixers with dummy data so @tf.function traces call() not __init__
        _dummy_qs    = tf.zeros((1, self.MAX_REQUESTS), dtype=tf.float32)
        _dummy_state = tf.zeros((1, mixer_state_dim),   dtype=tf.float32)
        self.mixer((_dummy_qs, _dummy_state))
        self.target_mixer((_dummy_qs, _dummy_state))

        # Optimizer specifically for QMIX training (trains both Agent and Mixer)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    def print_weight(self , model):
        for r in model.get_weights():
            print(r)
    def create_model(self):
        # Calculate the size of the flat input vector from your definition
        # (128 + SIZE + 2 * SIZE^2)
        flat_input_dim = self.OBSERVATION_SPACE_VALUES[0] * self.OBSERVATION_SPACE_VALUES[1]
        
        print('============== INITIALIZING GAT ADAPTER MODEL ==============')
        print(f'Input Dim: {flat_input_dim} -> Reshaping to: {self.SIZE} Nodes')
        
        # Initialize the GAT Model
        model = QRoutingGATFlat(
            num_nodes=self.SIZE, 
            input_flat_dim=flat_input_dim,
            hidden_dim=64,   # Internal feature size per node
            num_heads=4      # Number of attention heads
        )
        
        # Build the model by passing a dummy input (required for Subclassed models)
        # This initializes the weights so you can print summary or set weights
        dummy_input = tf.zeros((1, flat_input_dim))
        model(dummy_input)
        
        # Compile with your existing optimizer settings
        model.compile(loss="mse", optimizer=Adam(learning_rate=lr, clipvalue=clip_value), metrics=['accuracy'])
        
        return model
    
    def create_model_old(self):
        # try:
        #     model = load_model(self.model_name)
        #     print('=====================================================model loaded from ',self.model_name,' =====================================')
        #     # print(model.weights)
            
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
        # print('replay_memory size ' ,  len(replay_memory))
        # self.priorities.append(priority)
    




    # Trains main network every step during episode
    def get_last_n(self , d, n):
        """Return last n elements of a deque efficiently"""
        return list(islice(d, len(d)-n, len(d)))
    
    # Inside DQRLAgentDist class
    # --- COMPILED TRAINING STEP (Fast Execution) ---
    @tf.function
    def _train_step(self, padded_states, padded_actions, padded_next_states, global_states, next_global_states, rewards, dones, mask_tensor):
        with tf.GradientTape() as tape:
            # 1. Agent Forward Pass (ONLINE)
            # Reshape to (Batch * Max_Reqs, State_Dim) to feed to model efficiently
            flat_states = tf.reshape(padded_states, (-1, self.OBSERVATION_SPACE_VALUES[0]))
            
            # CRITICAL: calling self.model(...) tracks gradients
            all_q_values = self.model(flat_states) 
            
            # Reshape back to (Batch, Max_Reqs, Action_Space)
            all_q_values = tf.reshape(all_q_values, (MINIBATCH_SIZE, self.MAX_REQUESTS, -1))

            # 2. Gather Q-values for the specific actions taken
            # Indices setup for gather_nd
            action_indices = tf.cast(padded_actions, tf.int32)
            batch_indices = tf.expand_dims(tf.range(MINIBATCH_SIZE), 1) * tf.ones_like(action_indices)
            req_indices = tf.expand_dims(tf.range(self.MAX_REQUESTS), 0) * tf.ones_like(action_indices)
            
            # [Batch_Idx, Req_Idx, Action_Idx]
            gather_indices = tf.stack([batch_indices, req_indices, action_indices], axis=-1)
            
            chosen_qs = tf.gather_nd(all_q_values, gather_indices)
            
            # Zero out padding
            agent_qs_inputs = chosen_qs * mask_tensor

            # 3. Target Calculations (Double DQN)
            flat_next_states = tf.reshape(padded_next_states, (-1, self.OBSERVATION_SPACE_VALUES[0]))
            
            online_next_qs = self.model(flat_next_states) 
            target_next_qs = self.target_model(flat_next_states)
            
            online_next_qs = tf.reshape(online_next_qs, (MINIBATCH_SIZE, self.MAX_REQUESTS, -1))
            target_next_qs = tf.reshape(target_next_qs, (MINIBATCH_SIZE, self.MAX_REQUESTS, -1))
            
            best_actions = tf.argmax(online_next_qs, axis=2, output_type=tf.int32)
            
            gather_indices_next = tf.stack([batch_indices, req_indices, best_actions], axis=-1)
            target_qs_selected = tf.gather_nd(target_next_qs, gather_indices_next)
            
            target_agent_qs_inputs = target_qs_selected * mask_tensor

            # 4. Mixer Forward Pass
            q_tot_online = self.mixer((agent_qs_inputs, global_states))
            target_q_tot = self.target_mixer((target_agent_qs_inputs, next_global_states))

            # 5. Loss
            y_target = rewards + (GAMMA * target_q_tot * (1 - dones))
            loss = tf.keras.losses.MSE(y_target, q_tot_online)

        # 6. Apply Gradients
        variables = self.model.trainable_variables + self.mixer.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss

    # --- MAIN TRAIN FUNCTION (Optimized Data Prep) ---
    def train_qmix(self, terminal_state):
        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

        # 1. Pre-allocate NumPy Arrays (Speed Optimization)
        # Using float32 matches TensorFlow native type to avoid casting lag
        padded_states = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS, self.OBSERVATION_SPACE_VALUES[0]), dtype=np.float32)
        padded_actions = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS), dtype=np.int32)
        mask = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS), dtype=np.float32)
        padded_next_states = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS, self.OBSERVATION_SPACE_VALUES[0]), dtype=np.float32)
        
        # Determine global state dim dynamically from first sample
        g_dim = len(minibatch[0][4].flatten()) 
        global_states = np.zeros((MINIBATCH_SIZE, g_dim), dtype=np.float32)
        next_global_states = np.zeros((MINIBATCH_SIZE, g_dim), dtype=np.float32)
        
        rewards_batch = np.zeros((MINIBATCH_SIZE, 1), dtype=np.float32)
        dones_batch = np.zeros((MINIBATCH_SIZE, 1), dtype=np.float32)

        # 2. Fill Data
        for i, sample in enumerate(minibatch):
            ts_states, ts_actions, ts_rewards, ts_next_states, g_state, next_g_state, done = sample
            
            num_reqs = min(len(ts_states), self.MAX_REQUESTS)
            
            if num_reqs > 0:
                padded_states[i, :num_reqs] = np.array(ts_states, dtype=np.float32)[:num_reqs]
                padded_actions[i, :num_reqs] = np.array(ts_actions, dtype=np.int32)[:num_reqs]
                padded_next_states[i, :num_reqs] = np.array(ts_next_states, dtype=np.float32)[:num_reqs]
                mask[i, :num_reqs] = 1.0
            
            global_states[i] = g_state.flatten()
            next_global_states[i] = next_g_state.flatten()
            rewards_batch[i] = np.sum(ts_rewards)
            dones_batch[i] = done

        # 3. Run Compiled Training Step
        # No need for manual conversion, TF handles NumPy input efficiently in @tf.function
        loss = self._train_step(padded_states, padded_actions, padded_next_states, global_states, next_global_states, rewards_batch, dones_batch, mask)

        # 4. Update Target Networks
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_mixer.set_weights(self.mixer.get_weights())
            self.target_update_counter = 0
            # print(f"--- QMIX Loss: {loss:.4f} ---") # Optional Debug
        # gc.collect() # Only enable if memory is tight, slows down loop

    def train_qmix2(self, terminal_state):
        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

        # --- PRE-PROCESSING (Outside Tape) ---
        # We need to structure data so we can feed it to the model in one go
        
        # 1. Prepare padded batches
        # Shape: [Batch_Size, MAX_REQUESTS, State_Dim]
        padded_states = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS, self.OBSERVATION_SPACE_VALUES[0]))
        # Shape: [Batch_Size, MAX_REQUESTS] (Indices of actions taken)
        padded_actions = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS), dtype=np.int32)
        # Mask to remember which slots are real requests vs padding
        mask = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS), dtype=np.float32)
        
        # Target Network Data
        padded_next_states = np.zeros((MINIBATCH_SIZE, self.MAX_REQUESTS, self.OBSERVATION_SPACE_VALUES[0]))
        
        global_states = []
        next_global_states = []
        rewards_batch = []
        dones_batch = []
        t1 = time.time()
        for i, sample in enumerate(minibatch):
            ts_states, ts_actions, ts_rewards, ts_next_states, g_state, next_g_state, done = sample
            
            # Limit to MAX_REQUESTS to prevent overflow
            num_reqs = min(len(ts_states), self.MAX_REQUESTS)
            
            # Fill the padded arrays
            if num_reqs > 0:
                padded_states[i, :num_reqs] = np.array(ts_states)[:num_reqs]
                padded_actions[i, :num_reqs] = np.array(ts_actions)[:num_reqs]
                padded_next_states[i, :num_reqs] = np.array(ts_next_states)[:num_reqs]
                mask[i, :num_reqs] = 1.0
            
            global_states.append(g_state.flatten())
            next_global_states.append(next_g_state.flatten())
            rewards_batch.append(np.sum(ts_rewards))
            dones_batch.append(done)
        print('Pre-processing time: ', time.time() - t1)

        t1 = time.time()
        # Convert to Tensors
        padded_states = tf.convert_to_tensor(padded_states, dtype=tf.float32)
        padded_next_states = tf.convert_to_tensor(padded_next_states, dtype=tf.float32)
        global_states = tf.convert_to_tensor(global_states, dtype=tf.float32)
        next_global_states = tf.convert_to_tensor(next_global_states, dtype=tf.float32)
        rewards = tf.reshape(tf.convert_to_tensor(rewards_batch, dtype=tf.float32), (-1, 1))
        dones = tf.reshape(tf.convert_to_tensor(dones_batch, dtype=tf.float32), (-1, 1))
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
        print('Tensor conversion time: ', time.time() - t1)
        t2 = time.time()
        # --- TRAINING (Inside Tape) ---
        with tf.GradientTape() as tape:
            # 1. Agent Forward Pass (ONLINE)
            # We reshape to (Batch * Max_Reqs, State_Dim) to feed to model
            flat_states = tf.reshape(padded_states, (-1, self.OBSERVATION_SPACE_VALUES[0]))
            
            # CRITICAL: calling self.model(...) tracks gradients!
            t1 = time.time()
            all_q_values = self.model(flat_states) 
            print('Agent forward pass time: ', time.time() - t1)
            
            # Reshape back to (Batch, Max_Reqs, Action_Space)
            all_q_values = tf.reshape(all_q_values, (MINIBATCH_SIZE, self.MAX_REQUESTS, -1))

            # 2. Gather Q-values for the specific actions taken
            # We need to pick the specific value at 'padded_actions' index
            action_indices = tf.cast(padded_actions, tf.int32)
            # Create a batch index grid to pair with action indices
            batch_indices = tf.expand_dims(tf.range(MINIBATCH_SIZE), 1) * tf.ones_like(action_indices)
            req_indices = tf.expand_dims(tf.range(self.MAX_REQUESTS), 0) * tf.ones_like(action_indices)
            
            # Full indices: [Batch_Idx, Req_Idx, Action_Idx]
            gather_indices = tf.stack([batch_indices, req_indices, action_indices], axis=-1)
            
            # Extract Q-values
            chosen_qs = tf.gather_nd(all_q_values, gather_indices)
            
            # Zero out the padding using the mask
            agent_qs_inputs = chosen_qs * mask_tensor

            # 3. Target Calculations (Can be outside tape, but easier here)
            # (Note: stop_gradient is implied for target model, but good practice to be explicit)
            flat_next_states = tf.reshape(padded_next_states, (-1, self.OBSERVATION_SPACE_VALUES[0]))
            
            # Double DQN Logic
            t1 = time.time()
            online_next_qs = self.model(flat_next_states) # For selection
            print('Online next Qs pass time: ', time.time() - t1)
            t1 = time.time()
            target_next_qs = self.target_model(flat_next_states) # For evaluation
            print('Target next Qs pass time: ', time.time() - t1)
            
            # Reshape
            online_next_qs = tf.reshape(online_next_qs, (MINIBATCH_SIZE, self.MAX_REQUESTS, -1))
            target_next_qs = tf.reshape(target_next_qs, (MINIBATCH_SIZE, self.MAX_REQUESTS, -1))
            
            # Max action from Online
            best_actions = tf.argmax(online_next_qs, axis=2, output_type=tf.int32)
            
            # Gather value from Target
            gather_indices_next = tf.stack([batch_indices, req_indices, best_actions], axis=-1)
            target_qs_selected = tf.gather_nd(target_next_qs, gather_indices_next)
            
            # Mask Target
            target_agent_qs_inputs = target_qs_selected * mask_tensor

            # 4. Mixer Forward Pass
            q_tot_online = self.mixer((agent_qs_inputs, global_states))
            target_q_tot = self.target_mixer((target_agent_qs_inputs, next_global_states))

            # 5. Loss
            y_target = rewards + (GAMMA * target_q_tot * (1 - dones))
            t1 = time.time()
            loss = tf.keras.losses.MSE(y_target, q_tot_online)
            print('Loss calculation time: ', time.time() - t1)

        print('Total training step time inside tape: ', time.time() - t2)
        # 6. Gradients
        # Now variables includes the agent's weights!
        t1 = time.time()
        variables = self.model.trainable_variables + self.mixer.trainable_variables
        print('Variable gathering time: ', time.time() - t1)
        t1 = time.time()
        gradients = tape.gradient(loss, variables)
        print('Gradient calculation time: ', time.time() - t1)
        t1 = time.time()
        self.optimizer.apply_gradients(zip(gradients, variables))
        print('Optimizer apply gradients time: ', time.time() - t1)

        # Update Targets
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_mixer.set_weights(self.mixer.get_weights())
            self.target_update_counter = 0
    
    def train(self, terminal_state):
        global replay_memory
        t1 = time.time()
        # global model_lock
        # if model_lock is None:
        #     model_lock = multiprocessing.Lock()

        # Start training only if certain number of samples is already saved
        print('----------len(self.replay_memory)----------------', len(replay_memory))
        # print('----------size(self.replay memory)----------------', get_deep_size(replay_memory)/1024/1024 , 'MB')

        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        

        # Get a minibatch of random samples from memory replay table
        # priorities = np.array(self.priorities)
        # probabilities = priorities / priorities.sum()

        # indices = np.random.choice(len(self.replay_memory), MINIBATCH_SIZE, p=probabilities)
        # minibatch = [self.replay_memory[i] for i in indices]


        # last_half = self.get_last_n(replay_memory, MINIBATCH_SIZE // 2)

        # minibatch = random.sample(replay_memory, MINIBATCH_SIZE//2)
        # minibatch.extend(last_half)

        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

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

        del X, y, current_states, new_current_states
        del current_qs_list, future_qs_list, minibatch
        tf.keras.backend.clear_session()
        gc.collect()

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
        # Inference mode: always exploit (no exploration)
        if INFERENCE_MODE:
            return 0
        if timeSlot < START_EPSILON_DECAYING:
            return eps_start
        if timeSlot >= END_EPSILON_DECAYING:
            return 0
        epsilon = eps_start - eps_start  * (timeSlot / END_EPSILON_DECAYING)
        return max(0, epsilon)
    

    
    def learn_and_predict_next_req_node_single(self, reqIndex, ent_matrix , req_matrix,dist_matrix,timeSlot):
        if not INFERENCE_MODE and timeSlot > 0:
            if timeSlot not in self.loaded_ts:
                self.loaded_ts.add(timeSlot)
                try:
                    ml = time.time()
                    load_model_from_redis(self.model , self.model_name)
                    # self.model = load_model(self.model_name)
                    print('model loaded from ' , self.model_name , ' at timeSlot ' , timeSlot , ' time taken ' , time.time() - ml)
                    if timeSlot+1 % 500 == 0:
                        self.save_model()
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

    # ── Improvement 1: skip MHA when caller already computed it ────────────
    def predict_with_attn(self, reqIndex, ent_matrix, req_matrix, dist_matrix,
                          timeSlot, attn_encoded):
        """
        Single-request predict reusing a pre-computed attention tensor.
        Avoids redundant MHA computation when multiple requests share the same
        req_matrix within a timeslot.

        ``attn_encoded`` shape: (num_requests_padded, 64)
        """
        req = list(req_matrix[reqIndex][:6])
        req[3] = req_matrix[reqIndex + len(req_matrix) // 2]
        # Cast int fields (req_matrix is float32)
        req[0] = int(req[0]); req[1] = int(req[1])
        req[2] = int(req[2]); req[4] = int(req[4])
        if req[5]:
            return None

        ent_arr = np.array(ent_matrix)
        curr_emb = attn_encoded[reqIndex]                    # (64,) — pre-computed
        neighbor_embs = get_neighbor_embeddings(ent_arr, req[2])
        context_vec   = apply_neighbor_attention(curr_emb, neighbor_embs)

        local = np.zeros(self.SIZE, dtype=np.float32)
        local[req[2]] = 10.0
        local[req[1]] = 10.0

        current_state = np.concatenate([
            curr_emb, context_vec, np.array(local, dtype=np.float32),
            ent_arr.flatten(), np.array(dist_matrix).flatten()
        ]).astype(np.float32)

        qs   = self.get_qs(current_state)
        mask = self.get_mask_one_req_schedule_route(req, ent_matrix, req_matrix)
        valid_actions = np.where(mask == 1)[0]

        epsilon = self.get_epsilon_linear(timeSlot)
        if np.random.random() > epsilon:
            action = int(np.argmax(np.where(mask == 1, qs, -np.inf)))
        else:
            action = int(np.random.choice(valid_actions))

        return [current_state.tolist(), action]

    # ── Improvement 2: batched predict (1 MHA pass + 1 forward pass) ───────
    def batch_predict_all_requests(self, req_indices, ent_matrix, req_matrix,
                                   dist_matrix, timeSlot):
        """
        Process *all* active requests in one shot:
          • MHA attention computed ONCE for the shared req_matrix
          • Single batched model.predict() call for all N states

        Returns dict {reqIndex: [state_list, action_int]}.
        Skipped (done) requests are not included.
        """
        # 1. Optionally refresh model weights (training mode only; inference
        #    uses the frozen model loaded at startup).
        if not INFERENCE_MODE and timeSlot > 0 and timeSlot not in self.loaded_ts:
            self.loaded_ts.add(timeSlot)
            try:
                load_model_from_redis(self.model, self.model_name)
            except Exception:
                print('batch_predict: no model in Redis, using current weights')

        # 2. MHA ONCE for all requests
        req_tensor = get_request_embeddings(req_matrix)
        try:
            attn_encoded = apply_request_attention(req_tensor).numpy()  # (N_pad, 64)
        except Exception:
            attn_encoded = np.zeros((len(req_matrix), 64), dtype=np.float32)

        ent_arr   = np.array(ent_matrix,  dtype=np.float32)
        dist_arr  = np.array(dist_matrix, dtype=np.float32)
        ent_flat  = ent_arr.flatten()
        dist_flat = dist_arr.flatten()

        valid = []   # list of (req_idx, req_row, state_vector)
        for req_idx in req_indices:
            req = list(req_matrix[req_idx][:6])
            req[3] = req_matrix[req_idx + len(req_matrix) // 2]
            # Cast fields used as array indices to native int (req_matrix is float32)
            req[0] = int(req[0])   # src
            req[1] = int(req[1])   # dst
            req[2] = int(req[2])   # current_node_id
            req[4] = int(req[4])   # request index
            if req[5]:             # done flag — truthy float works fine
                continue
            curr_emb      = attn_encoded[req_idx]
            neighbor_embs = get_neighbor_embeddings(ent_arr, req[2])
            context_vec   = apply_neighbor_attention(curr_emb, neighbor_embs)

            local = np.zeros(self.SIZE, dtype=np.float32)
            local[req[2]] = 10.0
            local[req[1]] = 10.0

            state = np.concatenate([curr_emb, context_vec, local,
                                    ent_flat, dist_flat]).astype(np.float32)
            valid.append((req_idx, req, state))

        if not valid:
            return {}

        # 3. Single batched forward pass
        states_batch = np.array([v[2] for v in valid], dtype=np.float32)  # (N,20228)
        epsilon = self.get_epsilon_linear(timeSlot)
        if np.random.random() > epsilon:
            # Use model.predict with batch — faster than calling self.model N times
            qs_batch = self.model(states_batch, training=False).numpy()    # (N, SIZE)
        else:
            qs_batch = np.random.rand(len(valid), self.SIZE).astype(np.float32)

        # 4. Mask + argmax per request
        results = {}
        for i, (req_idx, req, state) in enumerate(valid):
            mask = self.get_mask_one_req_schedule_route(req, ent_matrix, req_matrix)
            qs   = qs_batch[i]
            action = int(np.argmax(np.where(mask == 1, qs, -np.inf)))
            results[req_idx] = [state.tolist(), action]

        return results

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
        
        
        # if True:
        #     for i in range(len(last_action_table)-1 , -1 , -1):
        #         t2 = time.time()
        #         (request , action , ts ,current_node_id, current_state , next_state ,mask ,  done,reward) = last_action_table[i]
                
        #         # req_id , next_node_id = self.decode_schdeule_route_action(action)
        #         # req.append(request)
        #         # print('before find reward time ')
        #         # reward = self.find_reward_routing(request  , timeSlot ,current_node_id , next_node_id)
        #         # print('after find reward time ' )
        #         # reward = self.env.find_reward_routing(request  , timeSlot ,current_node_id , action)
        #         # print((request[0].id , request[1].id) , reward)


        #         # if len(R):
        #         #     f = 0


        #         #     reward = reward * ALPHA + GAMMA * R[-1]

        #         #     reward /= pathlen
        #         #     # print((request[0].id , request[1].id) , reward)

        #         #     # R.append(reward)
                        
        #         # else:
        #         #     # reward = reward*ALPHA + numsuccessReq * GAMMA + avgFidelity* DELTA
        #         #     reward = reward*ALPHA + numsuccessReq * GAMMA 
        #         #     # reward = numsuccessReq
        #         #     reward /= pathlen
        #         #     R.append(reward)
        #         # reward = reward*ALPHA + numsuccessReq * GAMMA 
        #         reward = numsuccessReq

        #         # reward /=10
        #         total_reward += reward
        #         # print('get reward time ' , time.time() -t2)
        #         t3 = time.time()
        #         transition = ( current_state, action, reward, next_state,mask,  done)
        #         trans.append(transition)


        #         # print('update  replay memory time ' , time.time() -t3)
        # t4 = time.time()
        # print('before update replay memory time ' , time.time()-t4)
        # self.update_replay_memory(trans, numsuccessReq)

        # print('time for update memory ' , time.time()-t4)
        # # self.env.algo.topo.reward_routing = {}
        t5 = time.time()














        # Aggregators for the whole time slot
        ts_states = defaultdict(list)
        ts_actions = defaultdict(list)
        ts_rewards = defaultdict(list)
        ts_next_states = defaultdict(list)
        
        # We need a representation of the GLOBAL state. 
        # Currently your state is local-centric. 
        # For QMIX, you might pick the state of the first request or a dedicated global vector.
        # Let's assume we use the first request's state structure as the global proxy for now, 
        # but ideally, this should be the raw entanglement matrix flattened.
        global_state_proxy = defaultdict(list) 
        next_global_state_proxy = defaultdict(list)

        a_ids = set()

        for i in range(len(last_action_table)):
            (request, action, ts, current_node_id, current_state, next_state, mask, done, reward, global_state , next_global_state , a_id) = last_action_table[i]
            
            ts_states[a_id].append(current_state)
            ts_actions[a_id].append(action)
            ts_rewards[a_id].append(reward) # Or numsuccessReq
            ts_next_states[a_id].append(next_state)
            
            if a_id not in a_ids:
                a_ids.add(a_id)
                global_state_proxy[a_id] = global_state  # Should be purely global info
            next_global_state_proxy[a_id] = next_global_state

        # Construct the Joint Transition
        # (List of States, List of Actions, List of Rewards, List of Next States, Global State, Next Global State, Done)
        print('size of a_ids ' , len(a_ids) , a_ids)
        if len(a_ids) > 0:
            # Per-action avg fidelity proxy: fraction of done actions with success reward
            n_actions = sum(len(ts_rewards[aid]) for aid in a_ids)
            n_success_actions = sum(
                1 for aid in a_ids for r in ts_rewards[aid] if r >= 10
            )
            avg_fidelity = n_success_actions / max(n_actions, 1)

            for a_id in a_ids:
                # Paper Eq. 9: R = r·λ + N_success·μ + F_avg·ν
                raw_rewards = ts_rewards[a_id]
                scaled_rewards = [
                    r * REWARD_LAMBDA + numsuccessReq * REWARD_MU + avg_fidelity * REWARD_NU
                    for r in raw_rewards
                ]
                transition = (ts_states[a_id], ts_actions[a_id], scaled_rewards, ts_next_states[a_id], global_state_proxy[a_id], next_global_state_proxy[a_id], False)
                self.update_replay_memory(transition, numsuccessReq)

        self.train_qmix(False)
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
        if timeSlot % 1 == 0:
            st = time.time()
            save_model_to_redis(self.model, self.model_name)
            print('model saved to redis at time slot ' , timeSlot, 'time taken ' , time.time() - st)
            # Also persist to disk every 500 timeslots so inference can load it
            # even if Redis is restarted between training and inference phases.
            if timeSlot % 500 == 0 and timeSlot > 0:
                save_model_to_disk(self.model)
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

        # save_model_to_redis(self.model, self.model_name)
        self.model.save((self.model_name))
        # print(self.model.weights)
        # del self.model





class QMixer(Model):
    def __init__(self, n_agents, state_shape, embed_dim=32):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = embed_dim

        # Hypernetwork 1: Generates weights for 1st layer of mixing
        # Input: Global State -> Output: n_agents * embed_dim weights
        self.hyper_w1 = layers.Dense(n_agents * embed_dim)
        # Hypernetwork 1 Bias
        self.hyper_b1 = layers.Dense(embed_dim)

        # Hypernetwork 2: Generates weights for 2nd layer (final)
        # Input: Global State -> Output: embed_dim * 1 weights
        self.hyper_w2 = layers.Dense(embed_dim)
        
        # Hypernetwork 2 Bias (V(s))
        self.hyper_b2 = Sequential([
            layers.Dense(embed_dim, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs):
        # inputs is a tuple: (agent_qs, states)
        # agent_qs shape: [batch_size, n_agents] (Q values of selected actions)
        # states shape: [batch_size, state_dim]
        agent_qs, states = inputs
        
        batch_size = tf.shape(agent_qs)[0]

        # 1. First Layer Weights (Enforce Monotonicity with Abs)
        w1 = tf.abs(self.hyper_w1(states))
        w1 = tf.reshape(w1, (batch_size, self.n_agents, self.embed_dim))
        
        b1 = self.hyper_b1(states)
        b1 = tf.reshape(b1, (batch_size, 1, self.embed_dim))

        # 2. Reshape Agent Qs for Matmul
        agent_qs_reshaped = tf.reshape(agent_qs, (batch_size, 1, self.n_agents))

        # 3. First Hidden Layer calculation
        # (Batch, 1, Agents) * (Batch, Agents, Embed) -> (Batch, 1, Embed)
        hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1) + b1)

        # 4. Second Layer Weights
        w2 = tf.abs(self.hyper_w2(states))
        w2 = tf.reshape(w2, (batch_size, self.embed_dim, 1))

        # 5. Second Layer Bias
        b2 = self.hyper_b2(states)
        b2 = tf.reshape(b2, (batch_size, 1, 1))

        # 6. Final Q_tot
        y = tf.matmul(hidden, w2) + b2
        
        # Reshape to [batch_size, 1]
        q_tot = tf.reshape(y, (batch_size, 1))
        return q_tot
    
if __name__ == '__main__':
    agent = DQRLAgentDist()
    agent.initiate()
    
    
