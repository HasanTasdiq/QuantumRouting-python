import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
# sys.path.append("../../..")

# from RoutingEnv import RoutingEnv
from .RoutingEnv import RoutingEnv
from itertools import combinations
import random


#Hyperparameters
NUM_EPISODES = 2500
LEARNING_RATE = 0.1
DISCOUNT = 0.95

GAMMA = 0.99

class Agent():
    def __init__(self , algo):
        #Initializations
        self.env = RoutingEnv(algo)
        # env = gym.make('Breakout-v0')
        # env.reset()
        self.nA = 2
        self.dim = 2 #ball position x*width + ball pos y and player position
        # Init weight
        self.w = np.random.rand(self.dim, self.nA)

        # Keep stats for final print of graph
        self.episode_rewards = []
        self.q_table = {x:(random.random(), random.random()) for x in list(combinations([n.id for n in  self.env.algo.topo.nodes], 2)) }
        self.last_action_table = {}
        # print(self.q_table)

# extract_state checks the image extracts the state
# which consists of the x,y position of the ball and
# x position of the player
    def extract_state(self , I):
        I = self.preprocess(I)
        found_ball = False
        ball_pos = [0,0]
        for i in range(0,len(I) - 4): # the last 3 rows have the block
            # 13,14,15 row is a red tile row we skip this row as its the same color as the ball
            for j in range(0,len(I[i])):
                if I[i][j] == 114 and i == 14 :
                    continue
                if I[i][j] == 114 :
                    if I[i+1][j] == 114:
                        found_ball = True
                        ball_pos = [i,j]
                        if I[i+2][j] == 114:
                            found_ball = False
                            ball_pos = [0,0]
                            if I[i+3][j] == 114 and I[i+4][j] == 114:
                                found_ball = True
                                ball_pos = [i,j]
        player_pos = 0
        for i in range(len(I)-3,len(I)): # find the position of the player in the last 3 rows
            for j in range(0,len(I[i])-2): # 8 is the length of the player
                # print(I[i][j:j+8],end=" ")
                if(np.sum(I[i][j:j+2]) == 114*2):
                    player_pos = j+2
                    break
            # print()
        # print(ball_pos,player_pos)
        return [(ball_pos[0]*72+ball_pos[1])/(82*72),(player_pos)/72]

    def to_grayscale(self , img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        # print('downloadsample ' , img)
        return img[0][::2, ::2]
        # return tuple(item[::2, ::2] for item in img)


    # preprocess reduces the size, crops the borders and the score and gives just the playable part
    def preprocess(self , img):
        img = self.to_grayscale(self.downsample(img))
        img = img[9+7:(len(img)-7)] # 9 is the score part which we crop and 7 border size in top and bottom
        for i in range(4): # border size is 4
            img = np.delete(img,0,1)
            img = np.delete(img,len(img[0])-1,1)
        print('preprocess ' , img)
        return img

    def policy(self, state,w):
        # print('policy ' , w)
        # z = state.dot(w)
        z = np.array([[0.65371988, 0.43197503, 0.14576321, 0.64972099]])
        # print("z",z) 
        exp = np.exp(z/2)
        # print("ha",exp/np.sum(exp))
        return exp/np.sum(exp)

    # Vectorized softmax Jacobian
    def softmax_grad(self , softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    def run(self):
        fout = open("Breakout_ret_RL",'w')
        foutstep =open("Breakout_step_RL","w")
        for e in range(NUM_EPISODES):

            state = self.env.reset()
            # state = np.asarray(extract_state(state))[None,:]
            grads = []	
            rewards = []
            # Keep track of game score to print
            score = 0
            st =0
            while True:

                # Uncomment to see your model train in real time (slower)
                # env.render()
                # env.render()

                # Sample from policy and take action in environment
                probs = self.policy(state,self.w)
                # probs = [0.25 , 0.25 , 0.25 , 0.25]
                # print(state)
                # print(probs)
                action = np.random.choice(self.nA,p=probs[0])
                next_state,reward,done,_ = self.env.step(action)
                st+=1
                if(score==0 and done == True):
                    reward=-10
                # print(extract_state(next_state),end=" ")
                # next_state = np.asarray(extract_state(next_state))[None,:]

                # Compute gradient and save with reward in memory for our weight updates
                # print(action)
                dsoftmax = self.softmax_grad(probs)[action,:]
                dlog = dsoftmax / probs[0,action]
                # print('dlog ' , dlog)
                # grad = state.T.dot(dlog[None,:])
                grad = np.array([[0.65371988, 0.43197503, 0.14576321, 0.64972099]])

                grads.append(grad)
                rewards.append(reward)		

                score+=reward

                # Dont forget to update your old state to the new state
                state = next_state

                if done or score < -1000:
                    break

            # Weight update
            for i in range(len(grads)):

                # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
                self.w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** r) for t,r in enumerate(rewards[i:])])
            # Append for logging and print
            self.episode_rewards.append(score)
            fout.write(str(score))
            fout.write('\n')
            foutstep.write(str(st))
            foutstep.write('\n')
            print("EP: " + str(e) + " Score: " + str(score) + "         ",probs[0]) 
    def learn_and_predict(self):
        state = self.env.reset()
        for pair in state:
            probs = self.policy(state,self.w)
            action = np.argmax(self.q_table[(pair[0].id, pair[1].id)]) if (pair[0].id, pair[1].id) in self.q_table else np.argmax(self.q_table[(pair[1].id, pair[0].id)]) 
            self.last_action_table[pair] = action
            # print('llllll ', action)
            self.env.step(pair , action)

            # print(len(self.env.algo.topo.needLinksDict) , action)
    def update_reward(self):
        print('update reward: ' , self.env.algo.result.finishedRequestPerRound[-1])
        for pair in self.q_table:
            if not pair in self.last_action_table:
                continue
            n1 = self.env.algo.topo.nodes[pair[0]]
            n2 = self.env.algo.topo.nodes[pair[1]]
            action = self.last_action_table[pair]

            usedCount = 0
            used = False
            if (n1,n2) in  self.env.algo.topo.needLinksDict:
                usedCount = self.env.algo.topo.needLinksDict[(n1, n2)].count(self.env.algo.timeSlot)
                used = True
            elif (n2, n1) in  self.env.algo.topo.needLinksDict:
                usedCount = self.env.algo.topo.needLinksDict[(n2, n1)].count(self.env.algo.timeSlot)
                used = True
            
            if used:
                if usedCount > 0:
                    reward = 1
                else:
                    reward = -1
                max_future_q = np.max(self.q_table[pair])
                current_q = self.q_table[pair][action]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                self.q_table[pair][action] = new_q
        self.last_action_table = {}




        
            

            
            




    # def update_reward(self):
    #     next_state,reward,done,_ = self.env.step(action)
    #     dsoftmax = self.softmax_grad(probs)[action,:]
    #     dlog = dsoftmax / probs[0,action]
    #     print('dlog ' , dlog)
    #     grad = np.array([[0.65371988, 0.43197503, 0.14576321, 0.64972099]])



# fout.close()
# foutstep.close()
# plt.plot(np.arange(NUM_EPISODES),episode_rewards)
# plt.title("MountainCar a=0.000025")
# plt.show()
# env.close()
            
if __name__ == '__main__':
    agent = Agent()
    agent.run()