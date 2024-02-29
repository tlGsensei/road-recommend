import sys
from tqdm import tqdm
sys.path.append('../../')

import pandas as pd
import numpy as np
import torch
from torch import nn

class Seq(torch.nn.Module):
    # 这是一个基本的RNN模型，
    # 初始化时候需要输入维度，hidden维度，batch大小,gru层数
    # 以及emb字典的维度
    def __init__(self, input_size, hidden_size, batch_size, num_layers, n_items, n_users, emb_size):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 初始化gru层
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        # 初始化embedding
        self.item_embedding = nn.Embedding(n_items, emb_size)
        self.user_embedding = nn.Embedding(n_users, emb_size)
        # 先使用交叉熵损失
        self.loss_func = nn.CrossEntropyLoss()
        self.dense = nn.Linear(self.hidden_size, emb_size)
        self.act = nn.Softmax()

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
        item_seq_emb = self.item_embedding(item_id_list)
        gru_output,_ = self.gru(item_seq_emb)
        gru_output = self.dense(gru_output)
        # seq_output 是GRU输出的十个
        seq_output =  self.gather_indexes(gru_output, seq_len)
        total_output = torch.mul(seq_output, user_emb)
        # 是否需要经过激活函数
        total_output = self.act(total_output)
        item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0,1))
        return logits
        # loss = self.loss_func(logits, target)
        # return loss

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

    def predict2(self, uid, item_id_list, seq_len):
        # 用来预测next item
        # 输入为用户id，用户历史学习序列seq
        # 输出预测的下一个item
        user_emb = self.user_embedding(uid)
        item_seq_emb = self.item_embedding(item_id_list)
        gru_output,_ = self.gru(item_seq_emb.unsqueeze(0))
        gru_output = self.dense(gru_output)
        # seq_output 是GRU输出的十个
        seq_output =  self.gather_indexes(gru_output, seq_len)
        item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0,1))
        max_val, max_id = logits.max(1)
        # 返回最有可能的作为下一个点击预测
        return max_id.item()

    def path_generator(self, user_id, seq, target):
        # 用来根据 target 搜寻至 target 的路径
        # 传入参数为用户id，历史学习序列seq， 目标学习资源target
        # 输出为路径path学习资源id的序列
        seq_len = len(seq)
        # 此处长度与dataloader里面填充序列长度一致
        length = 10

        def change_seq(seq):
            seq_len = len(seq)
            if seq_len>10:
                seq = seq[seq_len-10:]
                seq_len = 10
            else:
                zeros = [0]*(length-seq_len)
                new_seq = seq + zeros
            return new_seq, seq_len

        new_seq, seq_len = change_seq(seq)
        path = []
        next_item = self.predict(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
        path.append(next_item+1)
        n_item = 1
        last_item = next_item
        while next_item != target:
            seq.append(next_item)
            new_seq, seq_len = change_seq(seq)
            next_item = self.predict(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
            if last_item == next_item:
                path.append(target)
                next_item = target
            else:
                path.append(next_item)
            # 如果一直找不到，避免死循环
            last_item = next_item
            if n_item == 8:
                next_item = target
                path.append(target)
        return path
    
    def path_generator_s(self, scene, user_id, seq, target):
        # 用来根据 target 以及所在场景scene（初始，复习，考前） 搜寻至 target 的路径
        # 传入参数为用户id，历史学习序列seq， 目标学习资源target
        # 输出为路径path学习资源id的序列
        seq_len = len(seq)
        # 此处长度与dataloader里面填充序列长度一致
        length = 10

        def change_seq(seq):
            seq_len = len(seq)
            if seq_len>10:
                seq = seq[seq_len-10:]
                seq_len = 10
            else:
                zeros = [0]*(length-seq_len)
                new_seq = seq + zeros
            return new_seq, seq_len

        new_seq, seq_len = change_seq(seq)
        path = []
        if scene == '初始':
            next_item = self.predict2(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
            path.append(next_item)
            n_item = 1
            last_item = next_item
            while next_item != target:
                seq.append(next_item)
                new_seq, seq_len = change_seq(seq)
                next_item = self.predict2(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
                if last_item == next_item:
                    path.append(target)
                    next_item = target
                else:
                    path.append(next_item)
                # 如果一直找不到，避免死循环
                last_item = next_item
                if n_item == 8:
                    next_item = target
                    path.append(target)
        elif scene == '复习':
            next_item = seq[0]
            path.append(next_item)
            n_item = 1
            last_item = next_item
            while next_item != target:
                seq.append(next_item)
                new_seq, seq_len = change_seq(seq)
                next_item = self.predict(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
                if last_item == next_item:
                    path.append(target)
                    next_item = target
                else:
                    path.append(next_item)
                # 如果一直找不到，避免死循环
                last_item = next_item
                if n_item == 8:
                    next_item = target
                    path.append(target)
        else:
            next_item = self.predict(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
            path.append(next_item)
            n_item = 1
            last_item = next_item
            while next_item != target:
                seq.append(next_item)
                new_seq, seq_len = change_seq(seq)
                next_item = self.predict(torch.tensor(user_id), torch.tensor(new_seq), torch.tensor(seq_len))
                if last_item == next_item:
                    path.append(target)
                    next_item = target
                else:
                    path.append(next_item)
                # 如果一直找不到，避免死循环
                last_item = next_item
                if n_item == 8:
                    next_item = target
                    path.append(target)
        return path

class SeqDataLoader(torch.utils.data.Dataset):
    # 读取数据
    # 能够从交互数据统计需要使用的序列信息，以及序列长度，最终预测等等
    def __init__(self, dataset_path):
        super().__init__()
        self.max_item_list_len = 10
        self.min_item_list_len = 3
        # 读入数据
        self.data = pd.read_csv(dataset_path, sep='\t').to_numpy()
        # 利用函数对数据进行处理
        self.interaction, self.rating = self._load_data(dataset_path)
        self.interaction.rename(columns={'user_id:token':'user_id','item_id:token':'item_id_list'}, inplace=True)
        self.uid = self.interaction['user_id'].to_numpy()
        self.item_id_list = self.interaction['item_id_list'].to_numpy()
        self.seq_len = self.interaction['seq_len'].to_numpy()
        self.target = self.interaction['pos_item'].to_numpy()

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, index):
        return self.uid[index], self.item_id_list[index], self.seq_len[index], self.target[index]

    def _load_data(self, dataset_path):
        # load data and preprocess it
        self.data = pd.read_csv(dataset_path, sep='\t')
        # 得到列名
        columns = self.data.columns
        # 对每一列数据进行空缺值处理
        for field in columns:
            self.data[field].fillna(value='', inplace=True)
        # 根据最大最小交互长度筛选用户、物品
        # user_iter_num = Counter(self.data['user_id'].values)
        # item_inter_num = Counter(self.data['item_id'].values)
        # 先对数据按照user和时间戳升序排列
        self.data.sort_values(by=['user_id:token','timestamp:float'], inplace=True, ascending=True)
        # 再根据用户聚合，得到交互序列和评分序列
        inter_seq = self.data.groupby('user_id:token').agg({'item_id:token':'unique'}).reset_index()
        # 数据清洗——需要filter一些交互过少的用户,得到保留的
        user_id = self.filter_user(inter_seq)
        # self.data = self.data[self.data['user_id:token']]
        # 对保留的数据，找到每个用户最后一个交互的作为test
        test_data = self.data.drop_duplicates(subset=['user_id:token'],keep='last')
        self.data = self.data[~self.data.index.isin(test_data.index)]
        # 得到过滤后的交互数据
        new_inter_seq = self.data.groupby('user_id:token').agg({'item_id:token':'unique'}).reset_index()
        rating_seq = self.data.groupby('user_id:token').agg({'rating:float':'unique'}).reset_index()
        new_inter_seq = new_inter_seq[new_inter_seq['user_id:token'].isin(user_id)]
        # 合并new_inter_seq 和 test_data，pos_item 列存放要预测的内容
        test_data = test_data[['user_id:token','item_id:token']]
        test_data.rename(columns={'item_id:token':'pos_item'}, inplace=True)
        new_inter_seq = pd.merge(new_inter_seq, test_data, on='user_id:token', how='inner')
        # 计算得到交互序列的长度
        new_inter_seq['seq_len'] = new_inter_seq['item_id:token'].apply(lambda x: len(x))
        # 再得到交互序列长度，以及将数据填充为固定长度list
        new_inter_seq['item_id:token'] = new_inter_seq['item_id:token'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))
        rating_seq['rating:float'] = rating_seq['rating:float'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))

        return new_inter_seq, rating_seq

    def fill_seq(self, x, length):
        # 此函数用来将序列补齐为统一长度
        n = len(x)
        if n>=length:
            return x[:length]
        else:
            a = [0]*(length-n)
            x.extend(a)
            return np.array(x)

    def filter_user(self, inter_seq):
        # 此函数用于过滤一些交互过少的用户
        inter_seq['seq_len'] = inter_seq['item_id:token'].apply(lambda x: len(x))
        user_id = inter_seq[inter_seq['seq_len']>=self.min_item_list_len]['user_id:token']
        return user_id



def train(path, device, BATCH_SIZE, saved_path):
    train_input = SeqDataLoader(path)

    traindata = torch.utils.data.DataLoader(
        dataset = train_input,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    model = Seq(input_size = 8, hidden_size = 3, batch_size = BATCH_SIZE, num_layers = 3, n_items = 10, n_users = 80000, emb_size = 8)
    # n_items 代表序列最长长度
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):

        model.train()
        all_train_loss = 0
        
        # for batch_idx, uid, item_id_list, seq_len, target in enumerate(tqdm(traindata)):
        # for batch_idx, data in enumerate(tqdm(traindata)):
        for batch_idx, data in enumerate(traindata):
            uid = data[0].to(device)
            item_id_list = data[1].to(device)
            seq_len = data[2].to(device)
            target = data[3].to(device)
            optimizer.zero_grad()
            output = model(uid, item_id_list, seq_len, target)
            loss = model.loss_func(output, target)
            all_train_loss += loss
            loss.backward()
            optimizer.step()
        if (epoch+1)//10 == 0:
            print("training:",epoch,'trainloss:',all_train_loss/(batch_idx+1))

    # 接口调用的服务器没有GPU，所以模型要存到cpu上
    torch.save(model.cpu().state_dict(), saved_path)

def load(path, user_id, seq, target):
    model = Seq(input_size = 8, hidden_size = 3, batch_size = 256, num_layers = 3, n_items = 10, n_users = 80000, emb_size = 8)
    model.load_state_dict(torch.load(path))
    model.eval()
    output = model.path_generator(user_id, seq, target)
    return output

def load_with_scene(path, scene, user_id, seq, target):
    model = Seq(input_size = 8, hidden_size = 3, batch_size = 256, num_layers = 3, n_items = 10, n_users = 80000, emb_size = 8)
    model.load_state_dict(torch.load(path))
    model.eval()
    output = model.path_generator_s(scene, user_id, seq, target)
    return output