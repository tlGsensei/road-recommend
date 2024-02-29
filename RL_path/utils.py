import random
import math

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


def count_ngram(candidate, references, n):
    count = 0
    # candidate中连续n个和references中连续n个的匹配次数
    can_seq = []
    for i in range(len(candidate)-n+1):
        temp = candidate[i:i+n]
        temp = [str(x) for x in temp]
        temp = ','.join(temp)
        can_seq.append(temp)
    can_seq = set(can_seq)
    # print(can_seq)
    for i in range(len(references)-n+1):
        temp2 = references[i:i+n]
        temp2 = [str(x) for x in temp2]
        temp2 = ','.join(temp2)
        # print(temp2)
        if temp2 in can_seq:
            count += 1
    c = len(candidate) - n + 1
    r = len(references) - n + 1
    if c < r:
        bp = math.exp(1-(float(r)/c))
    elif c == r:
        bp = 1
    else:
        bp = math.exp(1-(float(c)/r))
    # print("r:",r,"c:",c)
    pr = count/(max(r,c)+0.000001)
    return pr, bp


def BLEU(candidate, references):
    precisions = []
    s = min(len(candidate),len(references))
    for i in range(s):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
        # print('P'+str(i+1), ' = ',round(pr, 2),'bp:',bp)
    # print('BP = ',round(bp, 2))
    bleu = sum(precisions)/(len(precisions)+0.00001)
    return bleu


def LCS(text1, text2):
    m, n = len(text1), len(text2)
    dp = [0] * (n+1)

    for i in range(1, m+1):
        dp2 = [0] * (n+1)
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp2[j] = dp[j-1] + 1
            else:
                dp2[j] = max(dp[j], dp2[j-1])
        dp = dp2
    
    return dp[n]

def cal_p_r(rec_path, true_path):
    x = LCS(rec_path,true_path)
    r = x*1.0/len(true_path)
    p = x*1.0/len(rec_path)
    f = (2*p*r)/(p+r+0.0000001)
    return p, r, f