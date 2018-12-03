#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:40:04 2018

@author: muzhi
"""


from maze_env import Maze
import numpy as np

import matplotlib.pyplot as plt


"""

Net

"""


class table_qlearning:
    
    
    ##init
    def __init__(
            self, n_actions,
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon=0.9,
            
            ):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.Q = self.__build_q_table()
        self.observation = []
        
        self.episode = 0
        self.reward_his = []
       
        
    def __build_q_table(self):
        q_table = dict()
        return q_table
        
    ## add to sampling pool
    def add_to_samplingpool():
        pass;
    
    ## 
    def save_observation(self,ob):
        self.observation = ob
    
    
    ##e-greedy choose action
    def e_greedy_choose_action(self, stat):
        
        ## Q table
       
        
        if np.random.uniform() < self.epsilon:
            
            #if no this stat before, add to Q
            if str(stat)  not in  self.Q.keys():
                self.Q[str(stat)] = np.zeros(4)
           
            action_value = self.Q[str(stat)]
            
           
           
            action = np.random.choice((np.where(action_value==max(action_value)))[0])##here should prevent the equal fixed
        else:
            # random choose 
            action = np.random.randint(self.n_actions)
        
        return action
    
    ## update the Q(s,a)table
    def update(self):
       
        next_stat = str(self.observation[0])
        stat = str(self.observation[3])
        action = self.observation[4]
        reward = self.observation[1]
        q_s_a = self.Q[str(stat)][action]
        ##if q(s(t+1),a) is not discovered
        if str(next_stat) not in  self.Q.keys():
                self.Q[str(next_stat)] = np.zeros(4)
                
        self.Q[str(stat)][action] =  q_s_a + self.learning_rate * (
                reward + self.discount_factor * max(self.Q[next_stat]) - q_s_a
                )
    
    ##add every step reward to histroy
    def add_record(self, rewards_record, steps):
        self.reward_his.append([rewards_record,steps]) 
        

    
    def plot(self):
        pass;


"""

main function


"""

#next_stat, reward, done, stat, action = q_learning.observation
#Q = q_learning.Q


##
    
if __name__ == "__main__":
    
    N_episode = 50
    env = Maze() 
    q_learning = table_qlearning(
            n_actions = 4,
            epsilon=0.8,
            )
    
    
    for episode in range(N_episode):
        print("starting an new episode\n")
        stat = env.reset()
        i_step = 0
        R =[] 
        while True:
            i_step += 1
            env.render()
            
            action = q_learning.e_greedy_choose_action(stat)            
                    
            next_stat, reward, done = env.step(action)
            q_learning.save_observation([next_stat, reward, done, stat, action])
            q_learning.update()
            stat = next_stat
            
            ##add reward
            R.append(reward)
            if done:
                break;
        q_learning.add_record(sum(R), i_step)
            
    print("Done !!!")
    env.destroy()
    
    
    steps = [q_learning.reward_his[x][1] for x in range(episode) ]
    plt.plot(range(len(steps)), np.array(steps))











###

##random choose an action 
##
def random_action():
    for episode in range(20):
        print("epsiode")
        observation = env.reset()
        
        while True:
            
            print("step")
            env.render()
            
            action = np.random.randint(0,len(env.action_space))
                    
            observation_, reward, done = env.step(action)
            
            
            if done:
                break;
        
            
    print("Game Over")
    env.destroy()

