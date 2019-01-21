#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:03:06 2018

@author: muzhi
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import numpy as np
import os



"""
A2C
"""

### shared network 

def build_shared_network(X):
  # two convolutional layers
  conv1 = tf.layers.conv2d(
    inputs=X, filters=64, kernel_size=[4,4], padding="same",
      activation=tf.nn.relu, name="conv1")
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(
    pool1, filters=32,  kernel_size=[4,4], padding="same",
      activation=tf.nn.relu, name="conv2")
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Fully connected layer
  fc1 = tf.layers.dense(
    inputs=tf.layers.flatten(pool2),
    units=128,
    activation=tf.nn.relu,
    name="fc1")
  return fc1

#  # Fully connected layer
#build_shared_network(x)
#env = gym.make("Breakout-v0")
##X=env.reset()
#X = tf.to_float(X) / 255.0#normalize
#x = tf.reshape(X,[-1,210,160,3]) 
###atcor
class Actor(object):
    ## def network and params
    def __init__(self, sess, n_action=4, pic_shape=[210,160,3]):
        ## 
        self.sess = sess
        ##state dims 210,160,3
        self.pic_shape = pic_shape
        self.state = tf.placeholder(shape=[None, pic_shape[0], pic_shape[1], pic_shape[2]],
                                    dtype=tf.uint8, name="state_pic")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.td_error= tf.placeholder(shape=[None], dtype=tf.float32, name="td_error")
        ##state_pic to input data
        #X = self.state /255.0
        
        self.batch_size = tf.shape(self.state)[0]
        
        with tf.variable_scope("shared_net"):
            X = tf.cast(self.state, tf.float32)/ 255.0
            fc1 = build_shared_network(X)
           
        with tf.variable_scope("Actor_net"):
            self.logits = tf.layers.dense(inputs=fc1, units=n_action, activation=None)
            self.probs = tf.nn.softmax(self.logits) + 1e-8## make >0
            #self.pred = {}
            #here can add entropy to encourage exploration
            #self.entropy = -tf.reduce_sum(self.probs*tf.log())
            
            ##get action's index in prob matrix
            gather_indices = tf.range(self.batch_size)*tf.shape(self.probs)[1]+self.action
            
            
            ##picked action prob 
            self.picked_action_probs = tf.gather(tf.reshape(self.probs,[-1]), gather_indices)
            ##
            self.loss = tf.reduce_mean(-tf.log(self.picked_action_probs)*self.td_error, name="actor_loss")
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss)
       
    def choose_action(self, state):
        
        
        prob = self.sess.run(self.probs, {self.state: state})
        # 
        x = state.reshape(1,210,160,3)
        prob = actor.sess.run(actor.probs, {actor.state: x})
        return np.random.choice(np.arange(prob.shape[1]), p=prob.ravel())##choose by prob
        
        
    def update(self, state, action, td_error):
        feed_dict = {self.state:state, self.action:action.reshape(-1), self.td_error:td_error.reshape(-1)}
        loss = self.sess.run(self.train_op, feed_dict)
        
        #feed_dict = {actor.state:state.reshape(-1,210,160,3), actor.action:action.reshape(-1), actor.td_error:td_error.reshape(-1)}
        #loss = actor.sess.run(actor.train_op, feed_dict)
        return loss
        
        
    def exchange_net_params(self):
        pass
    
    
    

###critic
class Critic(object):
    ## def network and params
    def __init__(self, sess, pic_shape=[210,160,3], discount_factor=0.9, reuse=True):
        ## 
        self.sess = sess
        ##original state dims (210,160,3)
        self.pic_shape = pic_shape
        self.state = tf.placeholder(shape=[None, pic_shape[0], pic_shape[1], pic_shape[2]], 
                                    dtype=tf.uint8, name="state_pic")
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        self.next_value = tf.placeholder(shape=[None], dtype=tf.float32, name="next_value")
        
        ##state_pic to input data
        #X = self.state /255.0
        self.discount_factor = discount_factor
        self.batch_size = tf.shape(self.state)[0]
        
        with tf.variable_scope("shared_net", reuse=reuse):
            X = tf.cast(self.state, tf.float32)/ 255.0
            fc1 = build_shared_network(X)
        with tf.variable_scope("Critic_net"):
            self.value = tf.layers.dense(inputs=fc1, units=1, activation=None)
            self.td_error = self.reward + self.discount_factor*self.next_value - self.value            
            self.loss = tf.square(self.td_error, name="critic_loss")
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss)
    
    def update(self, state, reward, next_state):

        next_value = self.sess.run(self.value,{self.state: next_state.reshape(-1,210,160,3)}).reshape(-1)
        td_error ,_ = self.sess.run([self.td_error, self.train_op],
                                    {self.state:state,self.reward:np.array(reward).reshape(-1),
                                     self.next_value:next_value.reshape(-1)})
        
        
        #next_value = critic.sess.run(critic.value,{critic.state: next_state.reshape(1,210,160,3)}).reshape(-1)
        #td_error ,_ = critic.sess.run([critic.td_error, critic.train_op],
        #                            {critic.state:state.reshape(1,210,160,3),
        #                             critic.reward:np.array(reward).reshape(-1),
        #                            critic.next_value:next_value.reshape(-1)})
        
        return td_error
        
    def exchange_net_params(self):
        pass
    

## pic save


## pic batch 
        
    
## run main
        



##path to save pic
"""        
PATH_IMG = "/home/muzhi/RL/code/img"
if not os.path.exists(PATH_IMG):
    os.mkdir(PATH_IMG)        
"""


        
env = gym.make("Breakout-v0")

sess = tf.Session()
actor = Actor(sess)
critic = Critic(sess)

sess.run(tf.global_variables_initializer())


NEED_RENDER = False
MAX_EPISODE = 10
BATCH_SIZE = 20
reward_his = []

for i_episode in range(MAX_EPISODE):
    print(i_episode)
    state = env.reset()
    reward_track = []
    while True:
        if NEED_RENDER ==True :
            env.render()
        #env.step(1)
        #action is int number 0,1,2,3
        action = actor.choose_action(state.reshape(-1,210,160,3))
        
        next_state, reward, done, lives= env.step(action)
        reward_track.append(reward)
        
        td_error = critic.update(state.reshape(-1,210,160,3), reward, next_state.reshape(-1,210,160,3))
        
        actor.update(state.reshape(-1,210,160,3), action, td_error)
        
        if done: 
            print("\n The reward is : ",sum(reward_track))
            break
        state = next_state
        
    reward_his.append(reward_track)
env.close()
R = [np.sum(reward_his[i]) for i in range(len(reward_his))]
plt.plot(range(len(reward_his)), R)       







