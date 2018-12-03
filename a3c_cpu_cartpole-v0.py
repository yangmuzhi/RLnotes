#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:36:04 2018

@author: muzhi
"""



import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import numpy as np
import collections
import threading

"""
A3C
"""

"""
input:  state dims (1,4)
"""

class Actor(object):
    ## def network and params
    def __init__(self, sess, n_action=2, state_shape=4, n_hidden=200):
        ## 
        self.sess = sess
        ##state dims (1,4)
        self.state_shape = state_shape
        self.state = tf.placeholder(shape=[None, state_shape],
                                    dtype=tf.float32, name="state_pic")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.td_error= tf.placeholder(shape=[None], dtype=tf.float32, name="td_error")        
        self.batch_size = tf.shape(self.state)[0]
        self.n_hidden = n_hidden
        self.n_action = n_action
           
        with tf.variable_scope("Actor_net"):
            x = tf.layers.dense(inputs=self.state, units=self.n_hidden, activation= tf.nn.relu)
            self.logits = tf.layers.dense(inputs=x, units=self.n_action, activation=None)
            self.probs = tf.nn.softmax(self.logits) + 1e-8## make >0           
            ##get action's index in prob matrix
            gather_indices = tf.range(self.batch_size)*tf.shape(self.probs)[1]+self.action
            ##picked action prob 
            self.picked_action_probs = tf.gather(tf.reshape(self.probs,[-1]), gather_indices)
            ##
            self.loss = tf.reduce_mean(-tf.log(self.picked_action_probs)*self.td_error, name="actor_loss")
            self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
            self.train_op = self.optimizer.minimize(self.loss)
       
    def choose_action(self, state):
        
        state = np.array(state).reshape(-1,self.state_shape)
        self.batch_size = state.shape[0]
        prob = self.sess.run(self.probs, {self.state: state})

        #prob = actor.sess.run(actor.probs, {actor.state: state})
        return np.random.choice(np.arange(prob.shape[1]), p=prob.ravel())##choose by prob
        
    def update(self, state, action, td_error):
        
        state = np.array(state).reshape(-1,self.state_shape)
        action = np.array(action).reshape(-1)
        td_error = np.array(td_error).reshape(-1)
        self.batch_size = state.shape[0]
        feed_dict = {self.state:state,
                     self.action:action, self.td_error:td_error}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        #print("actor loss: ",loss)
      
        
    def exchange_net_params(self):
        pass
    
    
    

###critic
class Critic(object):
    ## def network and params
    def __init__(self, sess, state_shape=4, discount_factor=0.9, n_hidden=200):
        ## 
        self.sess = sess

        self.state_shape = state_shape
        self.state = tf.placeholder(shape=[None, state_shape], dtype=tf.float32, name="state_pic")
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        self.next_value = tf.placeholder(shape=[None], dtype=tf.float32, name="next_value")

        self.discount_factor = discount_factor
        self.batch_size = tf.shape(self.state)[0]
        #batch_size = tf.shape(state)[0]
        self.n_hidden = n_hidden
        with tf.variable_scope("Critic_net"):
            ## here!!! this should be check carefully
            x = tf.layers.dense(inputs=self.state, units=self.n_hidden, activation=tf.nn.relu)
            # x = tf.layers.dense(inputs=state, units=n_hidden, activation=tf.nn.relu)
            self.value = tf.reshape(tf.layers.dense(inputs=x, units=1, activation=None),[-1])
            ##
            self.td_error = tf.add(tf.add(self.reward, tf.multiply(self.discount_factor, self.next_value)),
                                     -self.value) 
            # td_error = tf.add(tf.add(reward, tf.multiply(discount_factor, next_value)),
                                  # -value) 
            #loss = tf.reduce_mean(tf.square(td_error), name="critic_loss")
            #td_error=tf.add(tf.add(critic.reward, tf.multiply(critic.discount_factor, critic.next_value)),
                                     #-critic.value) 
            self.loss = tf.reduce_mean(tf.square(self.td_error), name="critic_loss")
            #tf.reduce_mean(tf.square(td_error), name="critic_loss")
            self.optimizer = tf.train.RMSPropOptimizer(0.03, 0.9)
            self.train_op = self.optimizer.minimize(self.loss)
    
    def update(self, state, reward, next_state):
        state = np.array(state).reshape(-1,self.state_shape)
        reward = np.array(reward).reshape(-1)
        next_state = np.array(next_state).reshape(-1,self.state_shape)
        self.batch_size = state.shape[0]
        

        next_value = self.sess.run(self.value,{self.state: next_state}).reshape(-1)
        #print("next_value: ", next_value )
        td_error, _, loss = self.sess.run([self.td_error, self.train_op, self.loss],
                                    {self.state:state, 
                                     self.reward:reward,
                                     self.next_value:next_value})

        #print("critic loss: ", loss)
        
        
        """
        td_error =  critic.sess.run([ critic.td_error,  critic.train_op],
                                    { critic.state:np.array(state).reshape(- critic.batch_size,210,160,3), 
                                      critic.reward:np.array(reward).reshape(-1),
                                      critic.next_value:next_value.reshape(-1)})
       
        #td_error ,_ = critic.sess.run([critic.td_error, critic.train_op],
        #                            {critic.state:state.reshape(1,210,160,3),
        #                             critic.reward:np.array(reward).reshape(-1),
        #                            critic.next_value:next_value.reshape(-1)})
        """
        return np.array(td_error).reshape(-1)

    def exchange_net_params(self):
        pass
      
"""

New Worker

"""        
    


##
class Worker(object):
    def __init__(self, sess, name, env, actor, critic, discount_facor=0.9, num_step=50, max_episode=10, state_shape=4):
        
        ###def actor and critic
        self.name = name## name 
        self.env = env
        self.state = self.env.reset()
        self.Transition = collections.namedtuple("Transition", 
                                                 ["state", "action", "reward", "next_state", "done"])
        self.num_step = num_step
        self.batch_size = 0 #just define the batch_size
        self.sess = sess
        self.reward_his = []    ## whole reward in history
        self.reward_track = []  ##reward per epsiode
        self.i_episode = 0
        self.state_shape = 4
        self.max_episode = max_episode
        self.Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        

        with tf.variable_scope(name):
            
            self.actor = actor(self.sess)
            self.critic = critic(self.sess)   
            
        #self.copy_params_from_global
        
    def run(self):
            
        states,actions,next_states,dones,rewards = self.play_n_step(self.num_step) 
        #states,actions,next_states,dones,rewards = worker.play_n_step(10) 
        td_error = self.critic.update(states, rewards, next_states) 
        #td_error = worker.critic.update(states, rewards, next_states)                
        self.actor.update(states, actions, td_error)
        
        if (self.i_episode % 50 ==0)&(self.i_episode>0):
            self.plot_reward()
            

    def play_n_step(self,num_step):
        
        transitions = []
        for i in range(num_step):
            ##choose action
            state = self.state
            action = self.actor.choose_action(state.reshape(-1,self.state_shape))
            
            #action = worker.actor.choose_action(worker.state)
            #next_state, reward, done, info = worker.env.step(action)
            
            next_state, reward, done, info = self.env.step(action)
            self.reward_track.append(reward)
            transitions.append(self.Transition(state=state,action=action,
                                               reward=reward,next_state=next_state,done=done))

                        
            if done:          
                self.reward_his.append(self.reward_track)                
                print("\n Episode",self.i_episode,"has been done\n In this episode the reward is ", sum(self.reward_track))
                self.reward_track = []
                self.i_episode += 1
                self.state = self.env.reset()    
                break
            self.state = next_state
        
        
        reward = 0.0
        if not transitions[-1].done:
            #next_value = self.sess.run(self.value,{self.state: next_state}).reshape(-1)
            reward = self.critic.sess.run(self.critic.value,
                                          {self.critic.state:transitions[-1].state.reshape(-1,self.state_shape)}).reshape(-1)
        
       

        # Accumulate minibatch exmaples
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.critic.discount_factor * reward
      # Accumulate updates
            states.append(transition.state)
            actions.append(transition.action)           
            rewards.append(reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
        return states,actions,next_states,dones,rewards

    
    def plot_reward(self):
        
        R = [np.sum(self.reward_his[i]) for i in range(len(self.reward_his))]
        plt.figure()
        plt.plot(range(len(self.reward_his)), R)   
        plt.show()


        
###define the env, sess        ##
     
env = gym.make("CartPole-v0")

sess = tf.Session()

worker = Worker(sess, name="worker2", env=env, actor=Actor, critic=Critic)

sess.run(tf.global_variables_initializer())

MAX_EPISODE = 2000
while True: 
    
    worker.run()   
    if worker.i_episode >= MAX_EPISODE:
        break

#len(worker.reward_his[-1])
        
###play the game
import time 
state = env.reset()
i = 0
while True:
    time.sleep(0.1)
    env.render()
    action = worker.actor.choose_action(state)
    next_state, reward, done, info = env.step(action)

    print(reward)
    if done:
        env.close()
        break
    
