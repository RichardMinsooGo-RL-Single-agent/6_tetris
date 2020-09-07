import copy
import tensorflow as tf

import random
import numpy as np
import time, datetime
from collections import deque
import pylab
import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K

from tetris_pg_nodis import TetrisApp
env = TetrisApp()

game_name = "03_dqn_a_1"  # the name of the game being played for log files
model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

rows = 14
cols = 7

class DQN_agent:
    def __init__(self):

        # get size of state and action
        self.progress = " "
        # self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.action_space = [i for i in range(4*7)]    # 28 grouped action : board 7x4
        self.action_size = len(self.action_space)
        self.next_stone_size = 6
        self.state_size = (rows+1, cols, 1)
        
        # train time define
        self.training_time = 10*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0000625
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        # final value of epsilon
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.00001
        self.epsilon = self.epsilon_max
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.hidden1, self.hidden2 = 512, 512
        
        self.ep_trial_step = 500
        
        # Parameter for Experience Replay
        self.size_replay_memory = 50000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        # self.memory = PrioritizedReplayBuffer(100000, alpha=0.6) #1000000
        
        # Parameter for Target Network
        self.target_update_cycle = 1000
        
        # Parameters for network
        self.img_rows , self.img_cols = rows+1, cols
        self.img_channels = 1 #We stack 4 frames

        # create main model and target model
        self.model = self.build_model('network')
        self.target_model = self.build_model('target')
        
        # Define Tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        
    def preprocess(self, gameimage):
        copy_image = copy.deepcopy(gameimage)
        ret = [[0] * cols for _ in range(rows+1)]
        for i in range(rows+1):
            for j in range(cols):
                if copy_image[i][j] > 0:
                    ret[i][j] = 1
                else:
                    ret[i][j] = 0

        ret = sum(ret, [])
        return ret

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self, network_name):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(self.img_rows,self.img_cols,self.img_channels)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        # sample a minibatch to train on
        minibatch = random.sample(self.memory, self.batch_size)

        # Save the each batch data
        states      = np.array( [batch[0] for batch in minibatch])
        actions     = np.array( [batch[1] for batch in minibatch])
        rewards     = np.array( [batch[2] for batch in minibatch])
        next_states = np.array( [batch[3] for batch in minibatch])
        dones       = np.array( [batch[4] for batch in minibatch])
        
        X_batch = states    
        q_value          = self.model.predict_on_batch(states)
        tgt_q_value_next = self.target_model.predict_on_batch(next_states)
        
        y_array = rewards + self.discount_factor*(np.amax(tgt_q_value_next, axis=1))*(1-dones)
        ind = np.array([x for x in range(self.batch_size)])
        q_value[[ind], [actions]] = y_array

        # Decrease epsilon while training
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else :
            self.epsilon = self.epsilon_min
            
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(X_batch, q_value, epochs=1, verbose=0)

    # get action from model using epsilon-greedy policy
    def get_action(self, env, state):
        # choose an action epsilon greedily
        action_arr = np.zeros(self.action_size)
        action = 0
        
        if random.random() < self.epsilon:
            # print("----------Random Action----------")
            if env.stone_number(env.stone) == 1 :
                action = random.randrange(14)
            elif env.stone_number(env.stone) == 6 or env.stone_number(env.stone) == 7 :
                action = random.randrange(2)*7 + random.randrange(6)
            elif env.stone_number(env.stone) == 2 or env.stone_number(env.stone) == 3 or env.stone_number(env.stone) == 4 :
                action = random.randrange(4)*7 + random.randrange(6)
            elif env.stone_number(env.stone) == 5 :
                action = random.randrange(6)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            Q_value = self.model.predict(state)
            action = np.argmax(Q_value[0])
        action_arr[action] = 1
            
        return action_arr, action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
            
    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())
            
        # print(" Weights are copied!!")

    def save_model(self):
        # Save the variables to disk.
        self.model.save_weights(model_path+"/model.h5")
        save_object = (self.epsilon, self.episode, self.step)
        with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % model_path)

def main():
    
    agent = DQN_agent()
    
    # Initialize variables
    # Load the file if the saved file exists
    if os.path.isfile(model_path+"/model.h5"):
        agent.model.load_weights(model_path+"/model.h5")
        if os.path.isfile(model_path + '/epsilon_episode.pickle'):
            
            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.step = pickle.load(ggg)
            
        print('\n\n Variables are restored!')

    else:
        print('\n\n Variables are initialized!')
        agent.epsilon = agent.epsilon_max
        
    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    # initialize target model
    agent.Copy_Weights()

    while time.time() - start_time < agent.training_time:
        # Reset environment
        env.start_game()
        state = agent.preprocess(env.gameScreen)
        state = np.reshape(state, [rows+1, cols, 1])
        
        done = False
        agent.score = 0
        ep_step = 0
        
        while not done and env.total_clline < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1

            # Select action
            action_arr, action = agent.get_action(env, np.reshape(state, [1, agent.img_rows , agent.img_cols, 1]))
            
            # run the selected action and observe next state and reward
            reward, _ = env.step(action)

            next_state = agent.preprocess(env.gameScreen)
            next_state = np.reshape(next_state, [agent.img_rows , agent.img_cols, 1])
            
            # 게임이 끝났을 경우에 대해 보상 -2
            if env.gameover:
                done = True
                reward = -2.0                
            else:
                done = False

            # store the transition in memory
            # agent.memory.add(state, action, reward, next_state, done)
            agent.append_sample(state, action, reward, next_state, done)
            
            # update the old values
            state = next_state
            # only train if done observing
            if agent.progress == "Training":
                if agent.step % 4 == 0:
                    agent.train_model()
                if done or agent.step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()

            agent.score += reward

            if done or env.total_clline == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                    scores.append(env.total_clline)
                    episodes.append(agent.episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>8,d}'.format(agent.episode), '/ total_clline :{:>4d}'.format(env.total_clline), \
                      '/ time step :{:>7,d}'.format(agent.step),'/ status :', agent.progress, \
                      '/ epsilon :{:>1.4f}'.format(agent.epsilon),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break
    # Save model
    agent.save_model()
    
    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./save_graph/tetris_Nature2015.png")

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()

if __name__ == "__main__":
    main()
