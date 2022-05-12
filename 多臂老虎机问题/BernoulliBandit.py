import torch
import matplotlib.pyplot as plt
import numpy as np
#该文件是一个老虎机的相关模型部分
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = torch.zeros(self.bandit.K)   #记录每个老虎机被选择的次数
        self.regret = 0                            #记录单步的懊悔
        self.action = []                           #记录每步选择了哪个老虎机
        self.regrets = []                       #记录累积的懊悔列表

    def upgrade_regret(self, K): #K为选择哪个老虎机
        self.regret += float(self.bandit.best_prob)-float(self.bandit.probs[K])  #后悔等于理想最优奖励值的期望减去实际选择的奖励期望
        self.regrets.append(self.regret)       #记录累积后悔值

    def run_one_step(self):  #根据策略选择老虎机K
        raise NotImplemented

    def run(self,num_steps):         #进行K步收集相应的后悔资料和动作资料
        for _ in range (num_steps):
            K = self.run_one_step()
            self.counts[K]+=1
            self.action.append(K)
            self.upgrade_regret(K)

class BernoulliBandit:
    def __init__(self,K):
        self.K=K
        self.probs = torch.rand(K) # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
        self.best_idx = torch.argmax(self.probs) # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx] # 最大的获奖概率

    def step(self,k):
        if np.random.rand()<self.probs[k]:
            return 1
        else:
            return 0


def plot_results(solvers,solver_names): #solvers存储不同求解策略的列表,solver_names存储不同策略的名字
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label = solver_names[idx])

    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('{}-armed bandit'.format(solvers[0].bandit.K))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(1)# 设定随机种子,使实验具有可重复性
    K=10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成一个{}臂伯努利老虎机".format(K))
    print("获奖概率最大的拉杆为{}号,其获奖概率为{}".format(bandit_10_arm.best_idx,bandit_10_arm.best_prob))

    # 随机生成了一个10臂伯努利老虎机
    # 获奖概率最大的拉杆为1号,其获奖概率为0.7203


