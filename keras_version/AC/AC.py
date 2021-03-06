import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from utils.sample_buffer import Sampling_Pool
from .Actor import Actor
from .Critic import Critic
import tensorflow as tf
import keras.backend as K
import os

sampling_pool = Sampling_Pool()
class AC:
    """
    """
    def __init__(self, state_shape, n_action, net, model_path='model/a2c'):
        self.state_shape = state_shape
        self.n_action = n_action
        self.lr = 1e-4
        self.gamma = 0.9
        share_net = self.build_share_net(net)
        self.critic = Critic(self.state_shape,1,
                self.lr, share_net)
        self.actor = Actor(self.state_shape,self.n_action,
                self.lr, share_net)
        self.cum_r = []
        self.actor_update = self.actor.update()
        self.critic_update = self.critic.update()
        self.model_path = model_path
        global graph
        graph = tf.get_default_graph()


    def build_share_net(self, net):
        # 暂时使用一个简单的网络, 后面应该做一个net.py
        # actor 和 critic 共享一个net
        if  isinstance(self.state_shape, int):
            inputs = Input((self.state_shape,))
        else:
            inputs = Input(self.state_shape)
        x = net(inputs)

        return Model(inputs, x)

    def update(self, sampling_pool):
        state, reward, done, action, next_state = \
            sampling_pool.get_sample(shuffle=False)
        value = self.critic.value(state)
        next_value = self.critic.value(next_state)
        td_error = reward + self.gamma * next_value - value
        self.actor_update([state, action, td_error])
        self.critic_update([state, value])
        sampling_pool.clear()

    def train(self, env, episode, sampling_pool=sampling_pool):

        with graph.as_default():
            tqdm_e = tqdm(range(episode))
            for i in tqdm_e:
                state = env.reset()
                cum_r = 0
                done = False
                while not done:
                    state_newaxis = state[np.newaxis,:]
                    action = self.actor.explore(state_newaxis)
                    next_state,reward,done, _ = env.step(action)
                    action_onehot = to_categorical(action, self.n_action)
                    ob = (state, reward, done, action_onehot, next_state)
                    sampling_pool.add_to_buffer(ob)
                    state = next_state
                    cum_r += reward

                self.update(sampling_pool)
                self.cum_r.append(cum_r)
                tqdm_e.set_description("Score: " + str(cum_r))
                tqdm_e.refresh()
                if (i > 10000) &  (not(i % 10000)):
                    self.save_model(f"{i}-eps-.h5")
            self.save_model(f"final-{i}-eps-.h5")

    def save_model(self, save_name):
        path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.model.save(os.path.join(path, save_name))
