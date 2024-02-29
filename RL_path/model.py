import scipy
import torch
import torch.nn as nn
import sys,os
import torch.nn.functional as F
import datetime
import torch.optim as optim
from utils import *
import math
import random
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class path_env(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, n_users, emb_size, item_emb_path, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 初始化环境
        self.env = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.device = device
        # item embedding 使用 KG 上预训练的结果
        KGmodel = torch.load(item_emb_path)
        self.item_embedding = nn.Embedding.from_pretrained(KGmodel['ent_embeddings.weight'])
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.scene_embedding = nn.Embedding(4,emb_size)
        # 先使用交叉熵损失
        self.loss_func = nn.CrossEntropyLoss()
        self.dense = nn.Linear(self.hidden_size, emb_size)
        self.act = nn.Softmax()
        self.concat = nn.Linear(3*emb_size,emb_size)

    def set_for_user(self, uid, inter_seq, inter_len, scene, target, pred_seq, pred_len):
        self.uid = uid
        self.u_emb = self.user_embedding(uid)
        self.scene = self.scene_embedding(scene)
        self.inter_seq = inter_seq
        self.inter_len = inter_len if inter_len[0]<15 else torch.tensor([15]).to(self.device)
        self.target = target
        self.actual_seq = pred_seq
        self.pred_seq = []
        self.actual_len = pred_len if pred_len[0]<15 else torch.tensor([15]).to(self.device)

    def set_state(self):
        # 需要获取当前序列，当前学习场景向量
        # 返回计算好的状态向量
        item_seq_emb = self.item_embedding(self.inter_seq)
        gru_output,_ = self.env(item_seq_emb)
        gru_output = self.dense(gru_output)
        seq_output =  self.gather_indexes(gru_output, self.inter_len)
        # 拼接场景向量、用户向量、序列向量
        state = torch.cat((seq_output,self.u_emb),dim=1)
        # batch_size
        state = torch.cat((state, self.scene),dim=1)
        return state

    def step(self, action):
        '''根据动作，计算奖励函数、下一个状态
        '''
        # 需要获取当前序列，当前学习场景向量
        #self.interact_list.append(action)
        inter_seq = self.inter_seq
        if self.inter_len<15:
            inter_seq = inter_seq.scatter_(dim=1, index=self.inter_len.unsqueeze(1), src=action.unsqueeze(-1))
            self.inter_len += 1
        else:
            inter_seq = torch.roll(inter_seq,-1,1)
            inter_seq.scatter_(dim=1, index=self.inter_len.unsqueeze(1), src=action.unsqueeze(-1))
        self.inter_seq = inter_seq
        self.pred_seq.append(action)
        item_seq_emb = self.item_embedding(inter_seq)
        gru_output,_ = self.env(item_seq_emb)
        gru_output = self.dense(gru_output)
        seq_output =  self.gather_indexes(gru_output, self.inter_len)
        # 拼接场景向量、用户向量、序列向量
        next_state = torch.cat((seq_output,self.u_emb),dim=1)
        # batch_size
        #scene_emb = self.scene_embedding(torch.tensor([random.randint(0,3) for _ in range(len(next_state))]).to(self.device))
        next_state = torch.cat((next_state,self.scene),dim=1)
        # 奖励函数包括两个部分，一个是当前动作和target的相似度，一个是路径相似度
        '''
        path_reward = []
        for i in range(len(self.interact_list)):
            #path_reward.append(BLEU(self.interact_list[i][:self.interact_len[i]], self.item_id_list[i][:self.interact_len[i]]))
            path_reward.append(F.cosine_similarity(self.item_embedding(self.interact_list[i][:self.interact_len[i]]), self.item_embedding(self.item_id_list[i][:self.interact_len[i]])).mean().item())
        path_reward = torch.tensor(np.array(path_reward)).to(self.device)
        '''
        path_reward = BLEU(self.actual_seq[0][:self.actual_len[0]+1], self.pred_seq)
        '''
        if self.interact_len<=self.seq_len:
            unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(self.item_id_list[self.interact_len]),dim=0)
        else:
            unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(self.target),dim=0)
        '''
        # 如果超过长度，直接使用target，否则使用序列对应位置元素
        compare = self.target * (len(self.pred_seq)-1>self.actual_len) + torch.gather(self.actual_seq,1,self.actual_len.unsqueeze(-1)).squeeze(1)* (len(self.pred_seq)-1<=self.actual_len)
        unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(compare),dim=1)
        reward = path_reward + unit_reward
        #reward = path_reward
        #reward = unit_reward
        # 判断是否完成目标
        done = (action==self.target)
        info = "state updata with action"
        return next_state, reward, done, info

    def step2(self, action):
        '''根据动作，计算奖励函数、下一个状态
        '''
        # 需要获取当前序列，当前学习场景向量
        #self.interact_list.append(action)
        inter_seq = self.interact_list
        inter_seq = inter_seq.scatter_(dim=1, index=self.interact_len.unsqueeze(1), src=action.unsqueeze(-1))
        self.interact_len += 1
        self.interact_list = inter_seq
        '''
        if self.interact_len < self.seq_len:
            item_seq = self.interact_list + [0]*(self.seq_len-self.interact_len)
        else:
            item_seq = self.interact_list[self.interact_len-self.seq_len:]
        '''
        item_seq_emb = self.item_embedding(inter_seq)
        gru_output,_ = self.env(item_seq_emb)
        gru_output = self.dense(gru_output)
        seq_output =  self.gather_indexes(gru_output, self.seq_len)
        # 拼接场景向量、用户向量、序列向量
        next_state = torch.cat((seq_output,self.u_emb),dim=1)
        # batch_size
        # scene_emb = self.scene_embedding(torch.tensor([random.randint(0,3) for _ in range(len(next_state))]).to(self.device))
        scene_emb = self.scene_embedding(torch.tensor([0 for _ in range(len(next_state))]).to(self.device))
        next_state = torch.cat((next_state,scene_emb),dim=1)
        # 奖励函数包括两个部分，一个是当前动作和target的相似度，一个是路径相似度
        '''
        path_reward = []
        for i in range(len(self.interact_list)):
            #path_reward.append(BLEU(self.interact_list[i][:self.interact_len[i]], self.item_id_list[i][:self.interact_len[i]]))
            path_reward.append(F.cosine_similarity(self.item_embedding(self.interact_list[i][:self.interact_len[i]]), self.item_embedding(self.item_id_list[i][:self.interact_len[i]])).mean().item())
        path_reward = torch.tensor(np.array(path_reward)).to(self.device)
        '''
        path_reward = BLEU(self.interact_list, self.item_id_list)
        '''
        if self.interact_len<=self.seq_len:
            unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(self.item_id_list[self.interact_len]),dim=0)
        else:
            unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(self.target),dim=0)
        '''
        # 如果超过长度，直接使用target，否则使用序列对应位置元素
        compare = self.target * (self.interact_len>self.seq_len) + torch.gather(self.item_id_list,1,self.interact_len.unsqueeze(-1)).squeeze(1)
        unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(compare),dim=1)
        reward = path_reward + unit_reward
        # 判断是否完成目标
        done = (action==self.target)
        info = "state updata with action"
        # if not done:
        #     reward=0*reward
        return next_state, reward, done, info

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, uid, item_id_list, seq_len, target):
        # uid 存放用户id
        # item_id_list 存放用户交互过物品id
        # seq_len 存放有效交互长度
        # target 存放预测物品id
        user_emb = self.user_embedding(uid)
        # 现随机设置场景，后续改为根据序列判断
        scene_emb = self.scene_embedding(torch.tensor([random.randint(0,3) for _ in range(len(user_emb))]).to(self.device))
        item_seq_emb = self.item_embedding(item_id_list)
        gru_output,_ = self.env(item_seq_emb)
        gru_output = self.dense(gru_output)
        # seq_output 是GRU输出的十个
        seq_output =  self.gather_indexes(gru_output, seq_len) # batch_size * emb_size
        total_output = torch.cat((seq_output,user_emb),dim=1)
        total_output = torch.cat((total_output,scene_emb),dim=1)
        # total_output = torch.mul(seq_output, user_emb)
        total_output = self.concat(total_output)
        # 是否需要经过激活函数
        total_output = self.act(total_output) # batch_size * emb_size
        item_emb = self.item_embedding.weight
        logits = torch.matmul(total_output, item_emb.transpose(0,1))
        loss = self.loss_func(logits, target)
        return loss

    def predict(self, uid, item_id_list, seq_len):
        # 用来预测next item
        # 输入为用户id，用户历史学习序列seq
        # 输出预测的下一个item
        user_emb = self.user_embedding(uid)
        item_seq_emb = self.item_embedding(item_id_list)
        gru_output,_ = self.gru(item_seq_emb.unsqueeze(0))
        gru_output = self.dense(gru_output)
        # seq_output 是GRU输出的十个
        seq_output =  self.gather_indexes(gru_output, seq_len)
        total_output = torch.mul(seq_output, user_emb)
        # 是否需要经过激活函数
        total_output = self.act(total_output)
        item_emb = self.item_embedding.weight
        logits = torch.matmul(total_output, item_emb.transpose(0,1))
        max_val, max_id = logits.max(1)
        # 返回最有可能的作为下一个点击预测
        return max_id.item()

    def save(self, gru_path):
        torch.save(self.env.state_dict(), gru_path)

    def load(self, gru_path):
        self.env.load_state_dict(torch.load(gru_path))

class path_env_without_user_model(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, n_users, emb_size, item_emb_path, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 初始化环境
        self.env = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.device = device
        # item embedding 使用 KG 上预训练的结果
        KGmodel = torch.load(item_emb_path)
        self.item_embedding = nn.Embedding.from_pretrained(KGmodel['ent_embeddings.weight'])
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.scene_embedding = nn.Embedding(4,emb_size)
        # 先使用交叉熵损失
        self.loss_func = nn.CrossEntropyLoss()
        self.dense = nn.Linear(self.hidden_size, emb_size)
        self.act = nn.Softmax()
        self.concat = nn.Linear(3*emb_size,emb_size)

    def set_for_user(self, uid, inter_seq, inter_len, scene, target, pred_seq, pred_len):
        self.uid = uid
        self.u_emb = self.user_embedding(uid)
        self.scene = self.scene_embedding(scene)
        self.inter_seq = inter_seq
        self.inter_len = inter_len if inter_len[0]<15 else torch.tensor([15]).to(self.device)
        self.target = target
        self.actual_seq = pred_seq
        self.pred_seq = []
        self.actual_len = pred_len if pred_len[0]<15 else torch.tensor([15]).to(self.device)

    def set_state(self):
        # 需要获取当前序列，当前学习场景向量
        # 返回计算好的状态向量
        item_seq_emb = self.item_embedding(self.inter_seq)
        # gru_output,_ = self.env(item_seq_emb)
        # gru_output = self.dense(gru_output)
        # seq_output =  self.gather_indexes(gru_output, self.inter_len)
        seq_output = self.gather_indexes(item_seq_emb, self.inter_len)
        # 拼接场景向量、用户向量、序列向量
        state = torch.cat((seq_output,self.u_emb),dim=1)
        # batch_size
        state = torch.cat((state, self.scene),dim=1)
        return state

    def step(self, action):
        '''根据动作，计算奖励函数、下一个状态
        '''
        # 需要获取当前序列，当前学习场景向量
        #self.interact_list.append(action)
        inter_seq = self.inter_seq
        if self.inter_len<15:
            inter_seq = inter_seq.scatter_(dim=1, index=self.inter_len.unsqueeze(1), src=action.unsqueeze(-1))
            self.inter_len += 1
        else:
            inter_seq = torch.roll(inter_seq,-1,1)
            inter_seq.scatter_(dim=1, index=self.inter_len.unsqueeze(1), src=action.unsqueeze(-1))
        self.inter_seq = inter_seq
        self.pred_seq.append(action)
        item_seq_emb = self.item_embedding(inter_seq)
        # gru_output,_ = self.env(item_seq_emb)
        # gru_output = self.dense(gru_output)
        # seq_output =  self.gather_indexes(gru_output, self.inter_len)
        seq_output = self.gather_indexes(item_seq_emb, self.inter_len)
        # 拼接场景向量、用户向量、序列向量
        next_state = torch.cat((seq_output,self.u_emb),dim=1)
        # batch_size
        #scene_emb = self.scene_embedding(torch.tensor([random.randint(0,3) for _ in range(len(next_state))]).to(self.device))
        next_state = torch.cat((next_state,self.scene),dim=1)
        # 奖励函数包括两个部分，一个是当前动作和target的相似度，一个是路径相似度
        '''
        path_reward = []
        for i in range(len(self.interact_list)):
            #path_reward.append(BLEU(self.interact_list[i][:self.interact_len[i]], self.item_id_list[i][:self.interact_len[i]]))
            path_reward.append(F.cosine_similarity(self.item_embedding(self.interact_list[i][:self.interact_len[i]]), self.item_embedding(self.item_id_list[i][:self.interact_len[i]])).mean().item())
        path_reward = torch.tensor(np.array(path_reward)).to(self.device)
        '''
        path_reward = BLEU(self.actual_seq[0][:self.actual_len[0]+1], self.pred_seq)
        '''
        if self.interact_len<=self.seq_len:
            unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(self.item_id_list[self.interact_len]),dim=0)
        else:
            unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(self.target),dim=0)
        '''
        # 如果超过长度，直接使用target，否则使用序列对应位置元素
        compare = self.target * (len(self.pred_seq)-1>self.actual_len) + torch.gather(self.actual_seq,1,self.actual_len.unsqueeze(-1)).squeeze(1)* (len(self.pred_seq)-1<=self.actual_len)
        unit_reward = F.cosine_similarity(self.item_embedding(action),self.item_embedding(compare),dim=1)
        reward = path_reward + unit_reward
        #reward = path_reward
        #reward = unit_reward
        # 判断是否完成目标
        done = (action==self.target)
        info = "state updata with action"
        return next_state, reward, done, info

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, uid, item_id_list, seq_len, target):
        # uid 存放用户id
        # item_id_list 存放用户交互过物品id
        # seq_len 存放有效交互长度
        # target 存放预测物品id
        user_emb = self.user_embedding(uid)
        # 现随机设置场景，后续改为根据序列判断
        scene_emb = self.scene_embedding(torch.tensor([random.randint(0,3) for _ in range(len(user_emb))]).to(self.device))
        item_seq_emb = self.item_embedding(item_id_list)
        # gru_output,_ = self.env(item_seq_emb)
        # gru_output = self.dense(gru_output)
        # # seq_output 是GRU输出的十个
        # seq_output =  self.gather_indexes(gru_output, seq_len) # batch_size * emb_size
        seq_output = self.gather_indexes(item_seq_emb, seq_len)
        total_output = torch.cat((seq_output,user_emb),dim=1)
        total_output = torch.cat((total_output,scene_emb),dim=1)
        # total_output = torch.mul(seq_output, user_emb)
        total_output = self.concat(total_output)
        # 是否需要经过激活函数
        total_output = self.act(total_output) # batch_size * emb_size
        item_emb = self.item_embedding.weight
        logits = torch.matmul(total_output, item_emb.transpose(0,1))
        loss = self.loss_func(logits, target)
        return loss

    def predict(self, uid, item_id_list, seq_len):
        # 用来预测next item
        # 输入为用户id，用户历史学习序列seq
        # 输出预测的下一个item
        user_emb = self.user_embedding(uid)
        item_seq_emb = self.item_embedding(item_id_list)
        # gru_output,_ = self.gru(item_seq_emb.unsqueeze(0))
        # gru_output = self.dense(gru_output)
        # # seq_output 是GRU输出的十个
        # seq_output =  self.gather_indexes(gru_output, seq_len)
        seq_output = self.gather_indexes(item_seq_emb, seq_len)
        total_output = torch.mul(seq_output, user_emb)
        # 是否需要经过激活函数
        total_output = self.act(total_output)
        item_emb = self.item_embedding.weight
        logits = torch.matmul(total_output, item_emb.transpose(0,1))
        max_val, max_id = logits.max(1)
        # 返回最有可能的作为下一个点击预测
        return max_id.item()

    def save(self, gru_path):
        torch.save(self.env.state_dict(), gru_path)

    def load(self, gru_path):
        self.env.load_state_dict(torch.load(gru_path))

class DQNConfig:
    def __init__(self):
        self.algo = "DQN"  # 算法名称
        self.env = 'path' # 环境名称
        # 保存结果的路径
        self.result_path = curr_path+"/outputs/" + self.env + '/'+curr_time+'/results/'  
        # 保存模型的路径
        self.model_path = curr_path+"/outputs/" + self.env + '/'+curr_time+'/models/'  
        self.train_eps = 200 # 训练的回合数
        self.eval_eps = 30 # 测试的回合数
        self.gamma = 0.95 # 强化学习中的折扣因子
        self.epsilon_start = 0.90 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500 # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64 # mini-batch SGD中的批量大小
        self.target_update = 4 # 目标网络的更新频率
        # 检测GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.hidden_dim = 256  # hidden size of net

class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的特征数即环境的状态数
            output_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super(DQN, self).__init__()
        self.action_dim = action_dim  # 总的动作个数,对应item个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):
        '''选择动作
        在路径推荐时，这一步对应输入状态为一个表征向量，输出预测的下一个交互对象
        使用 policy net 完成 q值 计算，选择最大的
        需要给出一个batch的动作选择结果，如果已完成，则不用继续更新
        '''
        self.frame_idx += 1
        '''
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1] # 选择Q值最大的动作
        else:
            action = random.randrange(self.action_dim)
        '''
        q_values = self.policy_net(state)
        action = q_values.max(1)[1] # 选择Q值最大的动作
        return action

    def predict(self,state,inter):
        # 利用 policy net 完成 动作预测，区别是这里不使用e-greedy
        with torch.no_grad():
            #state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            #action = q_values.max(1)[1]
            q_values = q_values.sort(1,True) # 选择Q值最大的动作
            action = q_values.indices[0][0]
            i = 1
            while action in inter:
                action = q_values.indices[0][i]
                i += 1
        return action.unsqueeze(0)

    def update(self):
        if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 转为张量
        '''
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)
        '''
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values.to(torch.float32), expected_q_values.unsqueeze(1).to(torch.float32))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()  
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 

    def update_batch(self,state,action,next_state,reward,done):
        q_values = self.policy_net(state).gather(dim=1, index=action.unsqueeze(1)) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.target_net(next_state).max(1)[1].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward + self.gamma * next_q_values * (torch.ones(len(done)).to(self.device)*done)
        loss = nn.MSELoss()(q_values.to(torch.float32), expected_q_values.unsqueeze(1).to(torch.float32))  # 计算均方根损失
        # 优化更新模型
        # self.optimizer.zero_grad()  
        # loss.backward(retain_graph=True)
        # loss.backward()
        # for param in self.policy_net.parameters():  # clip防止梯度爆炸
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()
        return loss 

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)