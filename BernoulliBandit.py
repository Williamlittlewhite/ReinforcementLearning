import torch
from Solve import *
import matplotlib.pyplot as plt
import numpy as np
class BernoulliBandit:
    def __init__(self,K):
        self.K=K
        self.probs = torch.rand(K) # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
        self.best_idx = torch.argmax(self.probs) # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx] # 最大的获奖概率

    def step(self,k):
        if torch.rand(1)<self.probs[k]:
            return 1
        else:
            return 0

# if __name__ == '__main__':
#     torch.manual_seed(1)# 设定随机种子,使实验具有可重复性
#     K=10
#     bandit_10_arm = BernoulliBandit(K)
#     print("随机生成一个{}臂伯努利老虎机".format(K))
#     print("获奖概率最大的拉杆为{}号,其获奖概率为{}".format(bandit_10_arm.best_idx,bandit_10_arm.best_prob))
#
#     # 随机生成了一个10臂伯努利老虎机
#     # 获奖概率最大的拉杆为1号,其获奖概率为0.7203
def plot_results(solvers,solver_names): #solvers存储不同求解策略的列表,solver_names存储不同策略的名字
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label = solver_names[idx])

    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('{}-armed bandit'.format(solvers[0].bandit.K))
    plt.legend()
    plt.show()


class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,init_prob=0.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon                          #探索率
        self.estimates = torch.tensor([init_prob]*self.bandit.K)#记录每个动作的期望奖励估值


    def run_one_step(self):
        if torch.rand(1)<self.epsilon:             #小于则epsilon进行探索exploration
            # k = int(torch.randint(low=0,high=self.bandit.K,size=(1,)))这里不能这么用
            k = np.random.randint(0,self.bandit.K)
        else:                                       #大于则进行利用exploitation
            k = torch.argmax(self.estimates)

        r = self.bandit.step(k)      #得到奖励r_t  self.counts[k]+1是为了更新K对应的动作计数器
        self.estimates[k]+= 1.0/(self.counts[k]+1)*(r-self.estimates[k])#根据期望奖励的增量式更新公式去更新该动作对应的期望奖励
        return k


if __name__ == "__main__":
    torch.manual_seed(1)
    K=10
    bandit_10_arm = BernoulliBandit(K)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm,epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print("epsilon-贪婪算法累积懊悔:",epsilon_greedy_solver.regret)#返回的是最后的误差
    plot_results([epsilon_greedy_solver],["EpsilonGreedy"])

