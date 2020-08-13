# from tetris_pg_new import TetrisApp
from tetris_pg_nodis import TetrisApp

import tensorflow as tf
import os
import pickle
import sys
import random
import numpy as np
from collections import deque
import time

import copy
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Add, Conv2D, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
import keras.backend as K

from replay_buffer import PrioritizedReplayBuffer

file_name = "09_dueling_PER_inception"  # the name of the game being played for log files
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
        
        # get size of state and action
        self.progress = " "
        # self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.action_space = [i for i in range(4*7)]    # 28 grouped action : board 7x4
        self.action_size = len(self.action_space)
        self.next_stone_size = 6
        self.state_size = (rows+1, cols, 1)
        
        # train time define
        self.training_time = 3*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0000625
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.00001
        self.epsilon = self.epsilon_max
        
        self.episode = 0
        self.ep_trial_step = 500
        
        # Parameter for Experience Replay
        self.size_replay_memory = 50000
        self.batch_size = 64
        
        self.global_step = 0
        
        # Experience Replay 
        # self.memory = deque(maxlen=self.size_replay_memory)
        self.memory = PrioritizedReplayBuffer(100000, alpha=0.6) #1000000
        
        # Parameter for Target Network
        self.target_update_cycle = 1000

        # PER 선언 및 관련 hyper parameter입니다.
        self.beta = 0.4 # 0.4
        self.beta_max = 1.0
        self.beta_decay = 100000 #5000000
        
        self.prioritized_replay_eps = 0.00001

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # custom loss function을 따로 정의하여 학습에 사용합니다.
        self.model_updater = self.model_optimizer()
        
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
        
        # mixed 0: 35 x 35 x 256
        branch1x1   = Convolution2D(64, 1, 1, subsample=(1, 1), border_mode='same')(net2)

        branch5x5   = Convolution2D(48, 1, 1, subsample=(1, 1), border_mode='same')(net2)
        branch5x5_2 = Convolution2D(64, 1, 1, subsample=(1, 1), border_mode='same')(branch5x5)

        branch3x3   = Convolution2D(64, 1, 1, subsample=(1, 1), border_mode='same')(net2)
        branch3x3_2 = Convolution2D(96, 3, 3, subsample=(1, 1), border_mode='same')(branch3x3)
        branch3x3_3 = Convolution2D(96, 3, 3, subsample=(1, 1), border_mode='same')(branch3x3_2)

        branch_pool   = AveragePooling2D((3, 3),strides=(1, 1),padding='same')(net2)
        branch_pool_2 = Convolution2D(32, 1, 1, subsample=(1, 1), border_mode='same')(branch_pool)
        
        net_merge = concatenate([branch1x1, branch5x5_2, branch3x3_3, branch_pool_2], axis=3,name='merged_0')
        
        net3 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same')(net_merge)
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
        model.summary()
        
        return model

    # after some time interval update the target model to be same with model
    def CopyWeights(self):
        self.target_model.set_weights(self.model.get_weights())

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
        
    def model_optimizer(self):
        target = K.placeholder(shape=[None, self.action_size])
        weight = K.placeholder(shape=[None, ])        
        
        # hubber loss에 대한 코드입니다.        
        clip_delta = 1.0        
        pred = self.model.output        
        err = target - pred        
        cond = K.abs(err) < clip_delta        
        squared_loss = 0.5 * K.square(err)
        linear_loss = clip_delta * (K.abs(err) - 0.5 * clip_delta)        
        loss1 = tf.where(cond, squared_loss, linear_loss)        
        
        # 기존 hubber loss에 importance sampling ratio를 곱하는 형태의 PER loss를 정의합니다.
        weighted_loss = tf.multiply(tf.expand_dims(weight,-1),loss1)

        loss = K.mean(weighted_loss, axis=-1)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        
        train = K.function([self.model.input, target, weight], [err], updates=updates)

        return train
    
    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
        
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
            
    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        (states, actions, rewards, next_states, dones, weight, batch_idxes) = self.memory.sample(self.batch_size, beta=self.beta)
        
        q_value          = self.model.predict(states)
        q_value_next     = self.model.predict(next_states)
        tgt_q_value_next = self.target_model.predict(next_states)
        
        # Double DQN
        for i in range(self.batch_size):
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(q_value_next[i])
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * tgt_q_value_next[i][a]

        err = self.model_updater([states, q_value, weight])        
        err = np.reshape(err,[self.batch_size, self.action_size])
        new_priorities = np.abs(np.sum(err,axis=1)) + self.prioritized_replay_eps
        self.memory.update_priorities(batch_idxes, new_priorities)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        # beta < 1.0 이면 beta를 조금씩 증가
        if self.beta < self.beta_max:
            self.beta += (self.beta_max - 0.4)/self.beta_decay
        else:
            self.beta = 1.0

def main():
    agent = DQNagent()
    
    # saving and loading networks
    if os.path.isfile(model_path+"/model.h5"):
        agent.model.load_weights(model_path+"/model.h5")
        
        if os.path.isfile(model_path + '/append_sample.pickle'):                        
            with open(model_path + '/append_sample.pickle', 'rb') as f:
                agent.memory = pickle.load(f)

            with open(model_path + '/epsilon.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.global_step, agent.beta = pickle.load(ggg)
            print("\n\n Successfully loaded \n\n")
    else:
        agent.epsilon = agent.epsilon_max
        print("\n\n Could not find old network weights")
        
    env = TetrisApp()
    start_time = time.time()
    
    agent.CopyWeights()
    
    while time.time() - start_time < agent.training_time:
        done = False
        score = 0.0
        env.start_game()

        state = pre_processing(env.gameScreen)
        state = np.reshape(state, [rows+1, cols, 1])

        while not done and env.total_clline < agent.ep_trial_step:
            agent.global_step += 1
            
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"
            
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

            # store the transition in memory
            agent.memory.add(state, action, reward, next_state, float(done))
            
            # update the old values
            state = next_state            
            score += reward
            
            # only train if done observing
            if agent.progress == "Training":
                if agent.global_step % 4 == 0:
                    agent.train_model()
                if done or agent.global_step % agent.target_update_cycle == 0:
                    # copy q_net --> target_net
                    agent.CopyWeights()
                    
            if done or env.total_clline == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                print("episode :{:>8,d}".format(agent.episode), "/ total_clline :{:>4d}".format(env.total_clline), \
                      "/ global_step: {:>10,d}".format(agent.global_step), "/ epsilon :{:>2.6f}".format(agent.epsilon), \
                      "/ beta :{:1.5f}".format(agent.beta),"/ score :{:>4.3f}".format(score))
                break
            
    agent.model.save_weights(model_path+"/model.h5")
    with open(model_path + '/append_sample.pickle', 'wb') as f:
        pickle.dump(agent.memory, f)
        
    save_object = (agent.epsilon, agent.episode, agent.global_step, agent.beta )
    with open(model_path + '/epsilon.pickle', 'wb') as ggg:
        pickle.dump(save_object, ggg)
    print(" Model saved!! \n\n")
    
    fin_time = int(time.time() - start_time)
    print("  Elasped time :{:02d}:{:02d}:{:02d}".format(fin_time // 3600, (fin_time % 3600 // 60), fin_time % 60),"\n\n")
    sys.exit()

if __name__ == "__main__":
    main()
