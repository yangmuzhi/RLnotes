from A2C.A2C import A2C
import time
from threading import Thread
from multiprocessing import cpu_count

class A3C(A2C):
    def __init__(self, state_shape, n_action, n_threads=None):
        self.state_shape = state_shape
        self.n_action = n_action
        if n_threads == None:
            self.n_threads = cpu_count()
        super(A3C, self).__init__(state_shape,n_action)
        
    def trainAsy(self, env, episodes):
        """异步地使用A2C的train方法
        """
        envs = [env for i in range(self.n_threads)]
        threads = [Thread(target=self.train, args=(envs[i], episodes)) for i in range(self.n_threads)]
        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]
        
        
        
        
        
        