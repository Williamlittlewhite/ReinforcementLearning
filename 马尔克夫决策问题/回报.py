import numpy as np
np.random.seed(0)

#定义状态概率矩阵，每行代表其现在的状态，每列代表可转移状态，行和一定为1
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

rewards = [-1,-2,-2,10,1,0]     #代表每个状态得到的奖励
gamma = 0.5                  #折扣率 0表示只考虑当前 1表示考虑未来奖励

#给定序列计算回报的函数
def compute_return(start_index,chain,gamma):
    G=0
    for i in reversed(range(start_index,len(chain))):
        G = gamma*G+rewards[chain[i]-1]
    return G

def BellmanEquation(P,rewards,gamma,states_num):
    rewards = np.array(rewards).reshape((-1,1))
    Value = np.dot(np.linalg.inv(np.eye(states_num,states_num)-gamma*P),rewards)
    return Value

if __name__=='__main__':
    #一个状态序列 s1 s2 s3 s6
    chain = [1,2,3,6]
    start_index = 0
    G = compute_return(start_index,chain,gamma)
    print("本序列求得回报:%s"%G)
    Value = BellmanEquation(P,rewards,gamma,6)
    print("MRP中每个状态的价值为:",Value)