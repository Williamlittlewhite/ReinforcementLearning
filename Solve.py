import torch
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = torch.zeros(self.bandit.K)   #记录每个老虎机被选择的次数
        self.regret = 0                            #记录单步的懊悔
        self.action = []                           #记录每步选择了哪个老虎机
        self.regrets = []                       #记录累积的懊悔列表

    def upgrade_regret(self, K): #K为选择哪个老虎机
        self.regret += self.bandit.best_prob-self.bandit.probs[K]  #后悔等于理想最优奖励值的期望减去实际选择的奖励期望
        self.regrets.append(self.regret)       #记录累积后悔值

    def run_one_step(self):  #根据策略选择老虎机K
        raise NotImplemented

    def run(self,num_steps):         #进行K步收集相应的后悔资料和动作资料
        for _ in range (num_steps):
            K = self.run_one_step()
            self.counts[K]+=1
            self.action.append(K)
            self.upgrade_regret(K)

