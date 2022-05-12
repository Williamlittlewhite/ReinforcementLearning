import numpy as np
from BernoulliBandit import *

class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.bandit = bandit
        self.a = np.ones(self.bandit.K)  #每根拉杆获得奖励的次数
        self.b = np.ones(self.bandit.K)  #每根拉杆没获得奖励的次数

    def run_one_step(self):
        samples = np.random.beta(self.a,self.b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self.a[k] += r
        self.b[k] += 1-r
        return  k

if __name__ == '__main__':
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    np.random.seed(1)
    solver = ThompsonSampling(bandit_10_arm)
    solver.run(5000)
    print("汤普森采样累积懊悔:",solver.regret)
    plot_results([solver],["汤普森采样"])