#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: muzhi

“Classic” deep Q-learning algorithm (DQN) based on
            UCB reinforcement learning course
            playing cartpole
"""
import gym
import tensorflow as tf
import numpy as np
import os
import collections
import random

class Deep_QN():
    def __init__(self,
            sess,
            n_actions=2,
            learning_rate=1e-4,
            discount_factor=0.9,
            epsilon=0.9,
            batch_size = 256,
            memory_size = 20000):

            self.sess = sess
            self.epsilon = epsilon
            self.discount_factor = discount_factor
            self.n_actions = n_actions
            self.learning_rate = learning_rate
            self.observation = []
            self.action_space = list(range(n_actions))
            self.memory_size = memory_size
            self.episode = 0
            self.reward_his = []
            self.sampling_pool = []

            self.state_shape = 4
            self.n_hidden = n_hidden
            self.n_action = n_action
            self.__build_net()

    def __build_net(self):
        # 考虑tensor以batch的形式输入
        # net的输入为state
        self.state = tf.placeholder(shape=[None, state_shape],
                                    dtype=tf.float32, name="state_vec")
        # 输出为n_action 个 Q
        self.q_target = tf.placeholder(shape=[None, self.n_actions],
                                    dtype=tf.float32,name='Q_target')
        # batch size
        self.batch_size = tf.shape(self.state)[0]

        # 建立eval net
        with tf.variable_scope("eval_net"):
            x = tf.layers.dense(inputs=self.state, units=self.n_hidden, activation= tf.nn.relu, name="l1")
            self.q_eval = tf.layers.dense(inputs=x, units=self.n_action, activation=None, name='final')
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope("train_op")
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        # 建立target net
        self.state_ = tf.placeholder(shape=[None, state_shape],
                                    dtype=tf.float32, name="state_vec_")
        with tf.variable_scope('target_net'):
            x = tf.layers.dense(inputs=self.state_, units=self.n_hidden, activation= tf.nn.relu, name='l1')
            self.q_target_ = tf.layers.dense(inputs=x, units=self.n_action, activation=None, name='final')

    # 把 s r a d 储存起来 replay buffer
    def add_to_sampling_pool(self):
        # 把sampling pool的大小限制在一个大小区间里
        for len(self.sampling_pool) > self.memory_size:
            _ = self.sampling_pool.pop(0)
            self.sampling_pool.append(self.observation)
        else:
            self.sampling_pool.append(self.observation)

    ## e-greedy 选择action
    def e_greedy_choose_action(self,state):
        feed_dict = {self.state:state}
        action_value = self.sess.run(self.q_eval, feed_dict)

        if np.random.uniform() < self.epsilon:
            action =  int(np.argmax(action_value))
        else:
            # random choose
            action = random.choice(self.action_space)
        return action

    def update(self, state, reward, next_state):

        # 随机从replay buffer中选取 batchsize样本
        # sampling pool 是一个 list， 每一个元素是 由 state， reward， action, next_state构成
        state = np.array(state).reshape(-1,self.state_shape)
        reward = np.array(reward).reshpe(-1)
        next_state = np.array(next_state).reshape(-1,self.state_shape)
        feed_dict = {self.state:state}
        q_eval = self.sess.run(self.q_eval, {self.state:state})
        q_next = self.sess.run(self.q_eval, {self.state:next_state})
        q_s_a = self.sess.run(self., )
