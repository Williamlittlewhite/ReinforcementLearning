import numpy as np

from BernoulliBandit import *
class UCB(Solver):
    def __init__(self,bandit,init_probs=0.0,coef=0):
        super(UCB, self).__init__(bandit)
        self.estimates = np.array([init_probs]*self.bandit.K)
        self.coef = coef
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef* np.sqrt(np.log(self.total_count)/(2*(np.array(self.counts)+1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1./ (self.counts[k]+1)*(r-self.estimates[k])
        return k

if __name__ == '__main__':
    K=10
    bandit_10_arm = BernoulliBandit(K)
    np.random.seed(1)
    coef = 1
    UCB_solver = UCB(bandit_10_arm,coef=coef)
    UCB_solver.run(5000)
    print("上置信界算法的累积懊悔为:",UCB_solver.regret)
    plot_results([UCB_solver],["UCB_solver"])