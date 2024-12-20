import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras import Input
import gc


from collections import deque
import time
import random
import os
import psutil
from RoutingEnv import RoutingEnv      #for ubuntu
# from .RoutingEnv import RoutingEnv   #for mac

from objsize import get_deep_size
NUM_EPISODES = 2500
LEARNING_RATE = 0.1


GAMMA = 0.9
ALPHA = .2
BETA = -1

ENTANGLEMENT_LIFETIME = 10
# Exploration settings

EPSILON_ = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 2500
EPSILON_DECAY_VALUE = EPSILON_/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCOUNT = 0.9
REPLAY_MEMORY_SIZE = 20000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 512  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 50  # Terminal states (end of episodes)
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

class SchedulerAgent:
    def __init__(self ,algo , pid):
        print('++++++++++initiating DQRL agent for:' , algo.name)
        self.print_memory_usage()
        self.env = RoutingEnv(algo)
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE *2 +2,self.env.SIZE,)  
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE  + 3,self.env.SIZE,)  
        # self.OBSERVATION_SPACE_VALUES = (self.env.SIZE + 3 + self.env.algo.topo.numOfRequestPerRound,self.env.SIZE,)  
        self.OBSERVATION_SPACE_VALUES = (self.env.algo.topo.numOfRequestPerRound , 4*self.env.SIZE + self.env.SIZE*self.env.SIZE + self.env.SIZE,)  
        self.model_name = algo.name+'_'+ str(len(algo.topo.nodes)) +'_'+str(algo.topo.alpha) +'_'+str(algo.topo.q) +'_'+'ScheduleAgent.keras'

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        print('----------size(target_model)----------------', get_deep_size(self.target_model)/1000000)
        print('----------size(self Scheduler Agent)----------------', get_deep_size(self)/1000000)
        self.print_memory_usage()


        # print(self.model.get_weights())
        # self.print_weight(self.model)

        # print(self.target_model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.last_action_reward = []
        self.reqState_qs = {}



    def print_memory_usage(self):
        pid = os.getpid()

        # Create a Process object using the process ID
        process = psutil.Process(pid)

        # Get the memory information for the process
        memory_info = process.memory_info()

        # Get the resident set size (RSS) in bytes, which represents the actual physical memory used
        rss = memory_info.rss

        # Convert the RSS to megabytes (MB)
        rss_mb = rss / (1024 * 1024)

        print(f"Memory usage: {rss_mb:.2f} MB")
        return rss_mb
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
        print('~~~~~~~~after model.add(Flatten(input_shape = self.OBSERVATION_SPACE_VALUES)) ',get_deep_size(model)/1000000)
        self.print_memory_usage()
        gc.collect()
        # time.sleep(10)


        # model.add(Dense(self.env.SIZE * 20 , activation='relu'))
        model.add(Dense(600 , activation='relu'))
        print('~~~~~~~~after Dense(self.env.SIZE * 10 ',get_deep_size(model)/1000000)
        self.print_memory_usage()
        gc.collect()
        # time.sleep(10)


        # model.add(Dense(72 , activation='relu'))
        model.add(Dense(600 , activation='relu'))
        print('~~~~~~~~after Dense(self.env.SIZE * 5',get_deep_size(model)/1000000)
        gc.collect()
        # time.sleep(10)

        model.add(Dense(600 , activation='relu'))
        print('~~~~~~~~after Dense(self.env.SIZE * 3',get_deep_size(model)/1000000)
        gc.collect()


        # model.add(Conv2D(32, 3, activation="relu"))
        # model.add(Flatten)

        model.add(Dense(self.env.algo.topo.numOfRequestPerRound, activation='linear')) 
        print(model.summary)
        # print('------------------self.model.get_weights()-------------------')
        # print(model.get_weights())

        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.compile(loss="mse", optimizer=Adam(learning_rate = 0.001, clipvalue=0.01 ), metrics=['accuracy'])
        # model._make_predict_function()
        self.print_memory_usage()

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        print('----------len(self.replay_memory)----------------', len(self.replay_memory))
        print('----------size(self.replay_memory)----------------', get_deep_size(self.replay_memory)/1000000)
        # print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return        
        # if len(self.replay_memory) < MINIBATCH_SIZE:
        #     return

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
        # print('---------future qs --------- ')
        # print(future_qs_list)
        # future_qs_list = self.target_model.predict(new_current_states , verbose=0, batch_size=64)
        # print('---------future qs 2...  --------- ')
        # print(future_qs_list)

        X = []
        y = []
        # print(len(minibatch))

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done,qs,reqmask) in enumerate(minibatch):
            # print('---action-- ' , action)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # if len(qs) > 0:
            #     current_qs_list[index] = qs
            if True:
                # max_future_q = np.max(future_qs_list[index])

                max_future_q = self.env.max_future_q_schedule(reqmask , future_qs_list[index])
                
                # print('++++++++++++++÷++++++++++++++ ' , reward , max_future_q, current_qs_list[index][action])
                # print(qs)
                
                # if len(qs):
                #     qval = qs[action]
                # else:
                #     qval = 0
                qval = current_qs_list[index][action]
                # print('++++++++++++++++++++++++++++ ' , reward , max_future_q, current_qs_list[index][action] , qval)

                # new_q = (1-LEARNING_RATE)*qval+ LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                new_q = reward + DISCOUNT * max_future_q
                # print('===========================in train reward: ' , reward , 'newq:' , new_q)
            # else:
            #     new_q = reward
            # print(new_q)
            # print(current_state)

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # current_qs2 = self.get_qs(current_state)
            # print(current_qs)
            # print('--------- ')
            # print(qs)
            # print('--------- ')

            # print(current_qs2)

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


    def learn_and_predict_next_request_route(self,requests):

        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        current_state = self.env.schedule_route_state(requests  )
        qs = self.get_qs(current_state)
        if np.random.random() > EPSILON_:

            t2 = time.time()

            # next_node = np.argmax(self.get_qs(current_state))
            # next_node_index = np.argmax(self.env.schedule_qs(self.get_qs(current_state)))
            next_node_index = self.env.get_next_request_id(requests ,qs )
            print('+++++++++++++++ predict time ============ ' , (time.time()-t2) , 'sec')
            
        else:  
            # next_node = np.random.randint(0, self.env.SIZE)
            next_node_index = self.env.rand_request(requests)
        
        return current_state, next_node_index , qs

    def learn_and_predict_next_request(self , requests):

        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        current_state = self.env.schedule_state(requests  )
        qs = self.get_qs(current_state)
        if np.random.random() > EPSILON_:

            t2 = time.time()

            # next_node = np.argmax(self.get_qs(current_state))
            # next_node_index = np.argmax(self.env.schedule_qs(self.get_qs(current_state)))
            next_node_index = self.env.get_next_request_id(requests ,qs )
            print('+++++++++++++++ predict time ============ ' , (time.time()-t2) , 'sec')
            
        else:  
            # next_node = np.random.randint(0, self.env.SIZE)
            next_node_index = self.env.rand_request(requests)
        
        return current_state, next_node_index , qs
    
    def learn_and_predict_next_random_request(self , requests):
        global EPSILON_
        timeSlot = self.env.algo.timeSlot
        current_state = self.env.schedule_state(requests  )

        next_node_index = self.env.rand_request(requests)
        
        return current_state, next_node_index , []
    

    def updateAction2(self , sdPair , successReq):
        reqIndex = sdPair[4]
        el = next((x for x in self.last_action_reward if x[9] == reqIndex), None)
        el[6] = successReq

    def update_action(self , requests ,  action  , current_state , done = False , t =0 , successReq = 0, qval = [],reqmask = [],reqIndex = 0):
        global EPSILON_

        timeSlot = self.env.algo.timeSlot

        next_state = self.env.schedule_state(requests)

        if next_state is None:
            next_state = current_state
        # done = False

        self.last_action_reward.append([action , timeSlot , current_state , next_state ,  done , t , successReq , qval,reqmask , reqIndex])
        # for a in self.last_action_reward:
        #     print(a[0] , a[1],a[7] , a[8])
        # print('-------')
            # print('last_action_reward ' , timeSlot , reqmask , qval)



        # self.train(False , self.env.algo.timeSlot)

    
    def update_reward(self):
        print('update reward DQRA :::::::::::::::::::::::: ' , len(self.last_action_reward) )
        global EPSILON_
        t1 = time.time()
        l = len(self.last_action_reward)
        R = [0 for _ in range(l+1)]
        succ = []
        for i in range(l-1 , -1 , -1):
            action , timeSlot , current_state , next_state ,  done , t , successReq , qval, reqmask, reqIndex = self.last_action_reward[i]
            # print('reqmask ' , i , action  ,  reqmask)
            f = 0
            if done:
                f = 1

            reward = successReq * ALPHA + (self.env.algo.topo.numOfRequestPerRound - successReq)* BETA * f + GAMMA * R[t+1] * (1-f)
            R[t] = reward
            self.update_replay_memory((current_state, action, reward, next_state, done,qval,reqmask))
            succ.append(successReq)
        print('in update reward:' , timeSlot , 'R', R )
        print('in update reward:' , timeSlot , 'success ', succ )


        # for (action , timeSlot , current_state , next_state ,reward ,  done) in self.last_action_reward:
        #     self.update_replay_memory((current_state, action, reward, next_state, done))

        self.train(False , self.env.algo.timeSlot)

        self.last_action_reward = []
        print('update_reward done in ' , time.time() - t1 , 'seconds\n')
        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE
        print('epsilon ' , EPSILON_)


    def save_model(self):
        self.model.save((self.model_name))
