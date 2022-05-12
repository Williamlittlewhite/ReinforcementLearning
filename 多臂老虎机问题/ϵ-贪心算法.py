import torch
from BernoulliBandit import *
#探索率不随时间而发生变化的情况
class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,init_prob=0.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon                          #探索率
        self.estimates = np.array([init_prob]*self.bandit.K)#记录每个动作的期望奖励估值


    def run_one_step(self):
        # if torch.rand(1)<self.epsilon: tensor类型和数的类型不能进行比较，用法不正确
        if np.random.random()<self.epsilon:    #小于则epsilon进行探索exploration
            # k = int(torch.randint(low=0,high=self.bandit.K,size=(1,)))这里不能这么用
            k = np.random.randint(0,self.bandit.K)
        else:                                       #大于则进行利用exploitation
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)      #得到奖励r_t  self.counts[k]+1是为了更新K对应的动作计数器
        self.estimates[k]+= 1.0/(self.counts[k]+1)*(r-self.estimates[k])#根据期望奖励的增量式更新公式去更新该动作对应的期望奖励
        #注意公式中的k-1对应这里代码的k
        return k

#探索率随时间而发生衰减的情况
class DecayingEpsilonGreedy(Solver):
    def __init__(self,bandit,init_prob=0.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.K)#记录每个动作的期望奖励估值
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        # if torch.rand(1)<self.epsilon: tensor类型和数的类型不能进行比较，用法不正确
        if np.random.random()<1/self.total_count:    #小于则epsilon进行探索exploration
            # k = int(torch.randint(low=0,high=self.bandit.K,size=(1,)))这里不能这么用
            k = np.random.randint(0,self.bandit.K)
        else:                                       #大于则进行利用exploitation
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)      #得到奖励r_t  self.counts[k]+1是为了更新K对应的动作计数器
        self.estimates[k]+= 1.0/(self.counts[k]+1)*(r-self.estimates[k])#根据期望奖励的增量式更新公式去更新该动作对应的期望奖励
        #注意公式中的k-1对应这里代码的k
        return k

if __name__ == "__main__":
    torch.manual_seed(1)
    K=10
    bandit_10_arm = BernoulliBandit(K)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm,epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print("epsilon-贪婪算法累积懊悔:",epsilon_greedy_solver.regret)#返回的是最后的误差
    plot_results([epsilon_greedy_solver],["EpsilonGreedy"])


    # np.random.seed(1)
    # epsilon = [1e-4,0.01,0.1,0.25,0.5]
    # solvers = [EpsilonGreedy(bandit_10_arm,epsilon=e) for e in epsilon]
    # solvers_names = ["epsilon={}".format(e) for e in epsilon]
    # for solve in solvers:
    #     solve.run(5000)
    #
    # plot_results(solvers,solvers_names)

    np.random.seed(1)
    solver = DecayingEpsilonGreedy(bandit_10_arm)
    solver.run(5000)
    print("epsilon衰减-贪婪算法累积懊悔:",solver.regret)
    plot_results([solver],["DecayingEpsilonGreedy"])