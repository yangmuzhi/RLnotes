"""
normal policy gradient
can use important sampling
"""

import math
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Dense
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from utils.sample_buffer import Sampling_Pool

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

class PG:

    def __init__(self, env, state_shape, n_actions, net):

        self.env = env
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.net = net
        self.lr = 1e-4
        self.gamma = 0.9
        self.sample_pool = Sampling_Pool()
        self._build_network()
        self.reward_his = []

        # settings
        # tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):
        self.state = tf.placeholder(
                    shape=[None, *self.state_shape], dtype=tf.float32,
                    name="state")
        self.action = tf.placeholder(
                    shape=[None], dtype=tf.uint8,
                    name="action")
        self.G = tf.placeholder(
                    shape=[None], dtype=tf.float32,
                    name="G_n")
        x = self.net(self.state)
        self.act_prob = Dense(self.n_actions, activation="softmax")(x)

        with tf.name_scope('loss'):
    # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.act_prob, labels=self.tf_acts)
            neg_log_prob = tf.reduce_sum(
                    -tf.log(self.act_prob) * tf.one_hot(
                    self.action, self.n_actions),
                    axis=1)
        self.loss = tf.reduce_mean(neg_log_prob * self.G)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def act(self, state):
        # 选择一个action
        state = state.reshape(-1, *state.shape)
        action_prob = self.sess.run(self.act_prob,
                                    feed_dict={self.state: state}).squeeze()
        return np.random.choice(range(self.n_actions), p=action_prob)


    def _get_G(self, reward):
        g = 0
        G = np.zeros(len(reward))
        for i in reversed(range(len(reward))):
            g = self.gamma * g + reward[i]
            G[i] = g
        return G

    def train(self, episode=10):
        tqdm_e = tqdm(range(episode))
        for i in tqdm_e:
            d = False

            while not d:
                s = self.env.reset()
                a = self.act(s)
                s_next, r, d, info = self.env.step(a)
                # ob : state, reward, done, action, next_state
                self.sample_pool.add_to_buffer((s,r,d,a,s_next))

            # trains
            s, r, d, a, n = self.sample_pool.get_sample()
            self.sample_pool.clear()
            G = self._get_G(r)
            self.sess.run(self.train_op, feed_dict={
                        self.state: s,
                        self.action: a,
                        self.G: G
            })
            self.reward_his.append(r.sum())
            tqdm_e.set_description("Score: " + str(r.sum()))
            tqdm_e.refresh()
