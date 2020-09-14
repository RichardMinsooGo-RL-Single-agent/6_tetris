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

from keras.layers import Lambda, Input, Add, Conv2D, concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K

from tetris_pg_new import TetrisApp
env = TetrisApp()
import pygame 

from replay_buffer import PrioritizedReplayBuffer

game_name = "04_dqn"  # the name of the game being played for log files
model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

rows = 14
cols = 7

class DQN_agent:
    def __init__(self):
        
        # self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.action_space = [i for i in range(4*7)]    # 28 grouped action : board 7x4
        self.action_size = len(self.action_space)
        self.next_stone_size = 6
        self.state_size = (rows+1, cols, 1)
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0000625
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.hidden1, self.hidden2 = 512, 512
        
        self.ep_trial_step = 500
        
        # Parameters for network
        self.img_rows , self.img_cols = rows+1, cols
        self.img_channels = 1 #We stack 4 frames

        # create main model and target model
        self.model = self.build_model('network')
        
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
        # DQN        
        state = Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2],))        
        
        net1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', \
                             input_shape=(self.state_size[0], self.state_size[1], self.state_size[2],))(state) 
        net2 = Activation('relu')(net1)
        net3 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same')(net2)
        net4 = Activation('relu')(net3)
        net5 = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same')(net4)
        net6 = Activation('relu')(net5)
        net7 = Flatten()(net6)
        net8 = Dense(self.hidden1)(net7)
        net9 = Activation('relu')(net8)
        
        state_layer_1 = Dense(self.hidden2)(net9)

        tgt_output = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(state_layer_1)
        model = Model(inputs = state, outputs = tgt_output)
        
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        
    # get action from model using epsilon-greedy policy
    def get_action(self, env, state):
        # choose an action epsilon greedily
        action_arr = np.zeros(self.action_size)
        action = 0
        
        # Predict the reward value based on the given state
        state = np.float32(state)
        Q_value = self.model.predict(state)
        action = np.argmax(Q_value[0])
        action_arr[action] = 1
            
        return action_arr, action
def main():
    
    agent = DQN_agent()
    
    # Initialize variables
    # Load the file if the saved file exists
    if os.path.isfile(model_path+"/model.h5"):
        agent.model.load_weights(model_path+"/model.h5")
        
    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    pygame.init()
    while agent.episode < 5:

        env.start_game()
        state = agent.preprocess(env.gameScreen)
        state = np.reshape(state, [rows+1, cols, 1])
        
        done = False
        agent.score = 0
        ep_step = 0
        
        while not done and env.total_clline < agent.ep_trial_step:
            time.sleep(0.5)
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

            # update the old values
            state = next_state
            agent.score += reward

            if done or env.total_clline == agent.ep_trial_step:
                agent.episode += 1
                scores.append(env.total_clline)
                episodes.append(agent.episode)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>8,d}'.format(agent.episode), '/ total_clline :{:>4d}'.format(env.total_clline), \
                      '/ time step :{:>7,d}'.format(agent.step), '/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()

if __name__ == "__main__":
    main()
