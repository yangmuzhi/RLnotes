"""
Regret Matching:
working example of Rock-Paper-Scissors
"""

import numpy as np


"""
define RPS agent
"""
# np.random.choice(np.arange(3), p=np.array([0.2,0.3,0.5]))

class Agent:

    def __init__(self,  init_policy=None, fixed=False):

        self.num_actions = 3
        self.fixed = fixed
        if init_policy is not None:
            self.policy = init_policy

        self.cum_regrets = np.zeros(self.num_actions)

    def choose_action(self):


        if not self.fixed:
            tmp = np.zeros(self.num_actions)
            self.policy = np.zeros(self.num_actions)

            idx = self.cum_regrets >0
            tmp[idx] = self.cum_regrets[idx]
            tmp[[not i for i in idx]] = 0
            # print("tmp: ",tmp)
            if tmp.sum() > 0:
                self.policy = tmp / tmp.sum()
            else:
                self.policy = np.ones(self.num_actions) / self.num_actions

        # print("policy: ",self.policy)


            # # 如果 cum全是0,随即策略
# if (self.tmp == np.zeros_like(self.cum_regrets)).all():
#     self.policy = np.ones(self.num_actions) / self.num_actions
            #
            # #如果存在负值,该策略随机
            # else:
            #     print(self.cum_regrets)
            #     self.policy[self.cum_regrets < 0] = 1 / self.num_actions
            #     print(self.policy)
            #     idx = self.cum_regrets >= 0
            #     self.policy[idx] = self.cum_regrets[idx] / self.cum_regrets[idx].sum()
            #     self.policy = self.policy / self.policy.sum()
            #     print(self.policy)

        # policy 的概率之和必须为1

        assert np.rint(self.policy.sum()) == 1.0, "policy 的概率之和必须为1"

        return np.random.choice(np.arange(self.num_actions), p=self.policy)

    def regrets_matching(self, action, opp_action, U_fn):
        # Regret Matching

        regrets = self._compute_regrets(action, opp_action, U_fn)
        self.cum_regrets += regrets

    def _compute_regrets(self, action, opp_action, U_fn):
        regrets = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            regrets[i] = U_fn(agent_a=i, opp_a=opp_action) - U_fn(agent_a=action, opp_a=opp_action)
        return regrets


class RPS_Env:

    def __init__(self, agent0, agent1):

        self.agent0 = agent0
        self.agent1 = agent1
        self.u_table = np.array([[0, -1, 1],
                                 [1, 0, -1],
                                 [-1, 1, 0]])
        self.U_fn = self._build_u_fn()

    def _build_u_fn(self):
        """
        R: 0
        P: 1
        S: 2
        """
        def U_fn(agent_a, opp_a):

            return self.u_table[agent_a, opp_a]
        return U_fn

    def train(self, eps=10):
        for i in range(eps):
            a = self.agent0.choose_action()
            opp_a = self.agent1.choose_action()
            self.agent0.regrets_matching(action=a, opp_action=opp_a, U_fn=self.U_fn)
            print("policy:",self.agent0.policy)
            print("cum_regrets:",self.agent0.cum_regrets)



# test
agent0 = Agent()
agent1 = Agent(init_policy=np.array([0.3,0.3,0.4]), fixed=True)
env = RPS_Env(agent0, agent1)

# # env.agent0.policy
# env.agent0.cum_regrets
# env.agent1.policy
# env.agent1.policy
# env.agent0.cum_regrets
env.train(1000000)
# np.rint(env.agent1.policy.sum())
print
