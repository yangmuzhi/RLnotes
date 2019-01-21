#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:37:00 2018

@author: muzhi
"""


from maze_env import Maze
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class deep_qlearning:

    ##init
    def __init__(
            self, n_actions,
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon=0.9,
            batch_size = 32,
            memory_size = 2000
            ):

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.observation = []
        self.action_space = np.array([0, 1, 2, 3])
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.episode = 0
        self.reward_his = []
        self.sampling_pool = []
        ###### init net work
        self.__build_net()

    ##build the nn using pytorch
    def __build_net(self):
        # two net
        self.Q_eval_net = Net()
        self.Q_target_net = Net()
        self.optimizer = torch.optim.SGD(self.Q_eval_net.parameters(), lr=0.05)
        self.loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        self.loss_his = []

    ## add to sampling pool
    # the memory size should less than memory_size
    def add_to_sampling_pool(self):
        if len(self.sampling_pool) > self.memory_size:
            _ = self.sampling_pool.pop(0)
            self.sampling_pool.append(self.observation)
        else:
            self.sampling_pool.append(self.observation);

    ##
    def save_observation(self,ob):
        self.observation = ob
        self.add_to_sampling_pool()
    ##e-greedy choose action
    def e_greedy_choose_action(self,stat):

        stat_action = []
        ##each action in stat
        for a in self.action_space:
        #stat is np.array, action_value is list
            stat_action.append(list(np.append(stat,a)))
        action_value = self.Q_eval_net(torch.tensor(stat_action))

        if np.random.uniform() < self.epsilon:
            action =  np.float(action_value.argmax ()) ## prevent the elements when equal,  always fix
        else:
            # random choose
            action = np.float(np.random.randint(self.n_actions))

        return action
    ## update the Q(s,a)table
    def update(self):

        ##sampling from sampling pool
        #ob : St+1, Reward, done, St, action
        #random choice a batch
        ob_idx = np.random.choice(range(len(self.sampling_pool)), self.batch_size)
        ob = [self.sampling_pool[i] for i in ob_idx]

        #store the q(s,a),s,a pairs for trainning
        q_list = []
        s_a_list = []

        for i in range(len(ob)):

            next_stat = ob[i][0]
            reward = ob[i][1]
            done = ob[i][2]
            stat = ob[i][3]
            action = ob[i][4]

            stat_action = np.append(stat, action)
        ##next state action
            next_stat_action = []
            for a in self.action_space:
        #stat is np.array
        #action_value is list
                next_stat_action.append(list(np.append(next_stat,a)))

            q_s_a = self.Q_target_net(torch.tensor(list(stat_action)))
            q_s_a = q_s_a + self.learning_rate * (reward + self.discount_factor * max(
                    self.Q_target_net(torch.tensor(list(next_stat_action)))))

            q_list.append(list(q_s_a))
            s_a_list.append(list(stat_action))
        q_list = torch.tensor(q_list)
        s_a_list = torch.tensor(s_a_list)

        ##train step
        pred = self.Q_eval_net(s_a_list)
        loss = self.loss_func(pred, q_list)

        self.loss_his.append(loss)

        self.optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        self.optimizer.step()        # apply gradients

    def update_target_net(self):
        self.Q_target_net = self.Q_eval_net

    ##add every step reward to histroy
    def add_record(self, rewards_record, steps):
        self.reward_his.append([rewards_record,steps])


class Net(torch.nn.Module):

    def __init__(self, n_input=3, n_hidden_1=32, n_hidden_2 = 32, n_output=1):
        super(Net, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_input, n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)# hidden layer
        self.predict = torch.nn.Linear(n_hidden_2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


net = Net()
net(torch.tensor([0.5,0.5,1]))
###


if __name__ == "__main__":

    N_episode = 100
    env = Maze()
    q_learning = deep_qlearning(
            n_actions = 4,
            epsilon=0.8,
            batch_size = 50,
            memory_size = 2000
            )

    step = 0
    for episode in range(N_episode):
        step += 1
        print("starting an new episode\n")
        stat = env.reset()
        i_step = 0
        R =[]
        while True:
            i_step += 1
            env.render()

            action = q_learning.e_greedy_choose_action(stat)

            next_stat, reward, done = env.step(int(action))
            ##
            stat = next_stat
            q_learning.save_observation([next_stat, reward, done, stat, action])

            if (step > 20)and (i_step % 10 ==0):
                q_learning.update()
                print("learning ")
            if (step > 20)and (i_step %20 ==0):
                    q_learning.update_target_net()


            ##add reward
            R.append(reward)
            if done:
                break;
        q_learning.add_record(sum(R), i_step)


    print("Done !!!")
    env.destroy()


    steps = [q_learning.reward_his[x][1] for x in range(episode) ]

    plt.figure()
    plt1 = plt.subplot(121)
    plt1.plot(range(len(steps)), np.array(steps))
    plt2 = plt.subplot(122)
    plt2.plot(range(len(q_learning.loss_his)), q_learning.loss_his)
    plt.show()
