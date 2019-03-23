from A2C.A2C import A2C
import time
from threading import Thread
from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import gym
from utils.sample_buffer import Sampling_Pool

class A3C(A2C):
    def __init__(self, state_shape, n_action, net, n_threads=None, model_path='model/a3c'):
        self.state_shape = state_shape
        self.n_action = n_action
        if n_threads == None:
            self.n_threads = cpu_count()
        else:
            self.n_threads = n_threads

        super(A3C, self).__init__(state_shape,n_action,net,model_path=model_path) 

    def trainAsy(self, env_name, episodes, use_gym=True):
        """异步地使用A2C的train方法
        """
        # use_gym = False env_name 是一个没有实例化的类
        # 
        if use_gym:
            envs = [gym.make(env_name) for i in range(self.n_threads)]
        else:
            envs = [env_name() for i in range(self.n_threads)]
        
        sampling_pools = [Sampling_Pool() for i in range(self.n_threads)]
        threads = [Thread(target=self.train, args=(envs[i], episodes, sampling_pools[i])) for i in range(self.n_threads)]
        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]
