#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization.
目标loss由三部分组成：1. clipped policy loss；2. value loss；3. entropy。
policy loss需要计算当前policy和old policy在当前state上的ratio。
需要注意的是，state分布依赖于old policy。因此，计算中的old policy是一样的。

使用 ac style

参考CY LYJ 学长的rlpack
"""

class PPO:

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

    def discount(self, reward):
        """
        """
        discounted_reward, cumul_reward = np.zeros_like(reward), 0
        for t in reversed(range(0, len(reward))):
            cumul_reward = reward[t] + cumul_reward * self.gamma
            discounted_reward[t] = cumul_reward
        return discounted_reward

    def update(self, sampling_pool):
        state, reward, done, action, next_state = \
            sampling_pool.get_sample(shuffle=False)
        value = self.critic.value(state)


        discounted_reward = self.discount(reward)
        advantages = discounted_reward - value
        self.actor_update([state, action, advantages])
        self.critic_update([state, discounted_reward])
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
