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
        self.model = self._build_network()
        self.net = net
        self.lr = 1e-4
        self.gamma = 0.9
        self.sample_pool = Sampling_Pool()

        # settings
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.saver = tf.train.Saver(max_to_keep=20)
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)

    def _build_network(self):
        self.state = tf.placeholder(
                    shape=[None, *self.state_shape], dtype=tf.float32,
                    name="state")
        self.action = tf.placeholder(
                    shape=[None], dtype=tf.float32,
                    name="action")
        self.G = tf.placeholder(
                    shape=[None], dtype=tf.float32,
                    name="G_n")
        x = self.net(self.state)
        self.act_prob = Dense(self.n_actions, activation="softmax")(x)

    def act(self, state):
        # 选择一个action
        action_prob = self.sess.run(self.act_prob,
                                    feed_dict={self.state: state})
        return np.random.choice(len(self.n_actions), p=action_prob)

    def train_agent(self, obs):
        """
        obs: s, e_next, a, r, d
        """
        with tf.name_scope('loss'):
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.act_prob, labels=self.tf_acts)
            neg_log_prob = tf.reduce_sum(
                            -tf.log(self.act_prob) * tf.one_hot(
                            self.tf_action, self.n_actions),
                            axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.G)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _get_G(self, reward):
        g = 0
        G = np.zeros(len(reward))
        for i in reversed(range(len(reward))):
            g = self.gamma * g + reward[i]
            G[i] = g
        return G

    def train(self, episodes=10):
        d = False
        for i in range(episodes):
            while not d:
                s = self.env.reset()
                s_next, r, d, info = self.act(s)
                # ob : state, reward, done, action, next_state
                self.sample_pool.add_to_buffer((s,r,d,a,s_next))

            # train
            s, r, d, a, n = self.get_sample()
            G = self._get_G(r)
            self.sess.run(self.train_op, feed_dict={
                        self.state: s,
                        self.action: a,
                        self.G: G
            })
