# coding: utf-8
from tetris_pg_new import TetrisApp
import tensorflow as tf
import os
import random
import numpy as np
from collections import deque
import time

import copy
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Add, Conv2D, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import keras.backend as K

import pygame
 
from replay_buffer import PrioritizedReplayBuffer

file_name = "06_tetris_duelingdqn"  # the name of the game being played for log files
model_path = "save_model/" + file_name
graph_path = "save_graph/" + file_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

cols = 7
rows = 14
ret = [[0] * cols for _ in range(rows+1)]

def pre_processing(gameimage):
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

class DQNagent():
    def __init__(self):
        
        # self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.action_space = [i for i in range(4*7)]    # 28 grouped action : board 7x4
        self.action_size = len(self.action_space)
        self.next_stone_size = 6
        self.state_size = (rows+1, cols, 1)
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0000625
        self.discount_factor = 0.99
        
        self.epsilon = 0. #1.
        
        
        self.episode = 0
        self.ep_trial_step = 500

        # create main model and target model
        self.model = self.build_model()
        
        # Define Tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        # Dueling DQN        
        state = Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2],))        
        
        net1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', \
                             input_shape=(self.state_size[0], self.state_size[1], self.state_size[2],))(state) 
        net2 = Activation('relu')(net1)
        net3 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same')(net2)
        net4 = Activation('relu')(net3)
        net5 = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same')(net4)
        net6 = Activation('relu')(net5)
        net7 = Flatten()(net6)
        net8 = Dense(512)(net7)
        net9 = Activation('relu')(net8)
        
        state_layer_1 = Dense(512)(net9)
        action_layer_1 = Dense(512)(net9)

        v = Dense(1, activation='linear', kernel_initializer='he_uniform')(state_layer_1)
        v = Lambda(lambda v: tf.tile(v, [1, self.action_size]))(v)
        a = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(action_layer_1)
        a = Lambda(lambda a: a - tf.reduce_mean(a, axis=-1, keep_dims=True))(a)
        tgt_output = Add()([v, a])
        model = Model(inputs = state, outputs = tgt_output)
        
        model.compile(loss='mse',optimizer = Adam(lr = self.learning_rate))
        
        model.summary()
        
        return model
        
    def get_action(self, env, state):
        if np.random.rand() <= self.epsilon:
            if env.stone_number(env.stone) == 1 :
                return random.randrange(14)
            elif env.stone_number(env.stone) == 6 or env.stone_number(env.stone) == 7 :
                return random.randrange(2)*7 + random.randrange(6)
            elif env.stone_number(env.stone) == 2 or env.stone_number(env.stone) == 3 or env.stone_number(env.stone) == 4 :
                return random.randrange(4)*7 + random.randrange(6)
            elif env.stone_number(env.stone) == 5 :
                return random.randrange(6)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
def main():
    agent = DQNagent()
    
    # saving and loading networks
    if os.path.isfile(model_path+"/model.h5"):
        agent.model.load_weights(model_path+"/model.h5")
        
    env = TetrisApp()
    pygame.init()
    while agent.episode < 5:

        done = False
        score = 0.0
        env.start_game()

        state = pre_processing(env.gameScreen)
        state = np.reshape(state, [rows+1, cols, 1])

        while not done and env.total_clline < agent.ep_trial_step:
            time.sleep(0.5)
            action = agent.get_action(env, np.reshape(state, [1, rows+1, cols, 1]))
            
            # run the selected action and observe next state and reward
            reward, _ = env.step(action)

            # 게임이 끝났을 경우에 대해 보상 -2
            if env.gameover:
                done = True
                reward = -2.0                
            else:
                done = False

            next_state = pre_processing(env.gameScreen)
            next_state = np.reshape(next_state, [rows+1, cols, 1])

            # update the old values
            state = next_state            
            score += reward
            
            if done or env.total_clline == agent.ep_trial_step:
                agent.episode += 1
                print("episode :{:>8,d}".format(agent.episode), "/ total_clline :{:>4d}".format(env.total_clline), \
                      "/ score :{:>4.3f}".format(score))
                break

if __name__ == "__main__":
    main()
