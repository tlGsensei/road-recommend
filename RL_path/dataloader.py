import random
import sys
from tqdm import tqdm
sys.path.append('../../')

import pandas as pd
import numpy as np
import torch
from torch import nn

class SeqDataLoader(torch.utils.data.Dataset):
    # 读取数据
    # 能够从交互数据统计需要使用的序列信息，以及序列长度，最终预测等等
    def __init__(self, dataset_path):
        super().__init__()
        self.max_item_list_len = 100
        self.min_item_list_len = 3
        # 读入数据
        self.data = pd.read_csv(dataset_path, sep='\t').to_numpy()
        # 利用函数对数据进行处理
        self.interaction, self.rating = self._load_data(dataset_path)
        self.interaction.rename(columns={'userid':'user_id','itemid':'item_id_list'}, inplace=True)
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
        columns = ['userid', 'itemid', 'behavior', 'timestamp']
        self.data = pd.read_csv(dataset_path, sep='\t', names=columns)
        # 对每一列数据进行空缺值处理
        for field in columns:
            self.data[field].fillna(value='', inplace=True)
        # 先对数据按照user和时间戳升序排列
        self.data.sort_values(by=['userid','timestamp'], inplace=True, ascending=True)
        self.data.drop_duplicates(subset=['userid','itemid'], keep='first', inplace=True)
        # 再根据用户聚合，得到交互序列和评分序列
        inter_seq = self.data.groupby('userid').agg({'itemid':'unique'}).reset_index()
        # 数据清洗——需要filter一些交互过少的用户,得到保留的
        user_id = self.filter_user(inter_seq)
        # 对保留的数据，找到每个用户最后一个交互的作为test
        test_data = self.data.drop_duplicates(subset=['userid'],keep='last')
        self.data = self.data[~self.data.index.isin(test_data.index)]
        # 得到过滤后的交互数据
        new_inter_seq = self.data.groupby('userid').agg({'itemid':'unique'}).reset_index()
        behave_seq = self.data.groupby('userid').agg({'behavior':'unique'}).reset_index()
        new_inter_seq = new_inter_seq[new_inter_seq['userid'].isin(user_id)]
        # 合并new_inter_seq 和 test_data，pos_item 列存放要预测的内容
        test_data = test_data[['userid','itemid']]
        test_data.rename(columns={'itemid':'pos_item'}, inplace=True)
        new_inter_seq = pd.merge(new_inter_seq, test_data, on='userid', how='inner')
        # 计算得到交互序列的长度
        new_inter_seq['seq_len'] = new_inter_seq['itemid'].apply(lambda x: len(x))
        behave_seq['seq_len'] = behave_seq['behavior'].apply(lambda x: len(x))
        # 再得到交互序列长度，以及将数据填充为固定长度list
        new_inter_seq['itemid'] = new_inter_seq['itemid'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))
        behave_seq['behavior'] = behave_seq['behavior'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))

        return new_inter_seq, behave_seq

    def fill_seq(self, x, length):
        # 此函数用来将序列补齐为统一长度
        n = len(x)
        if n>=length:
            return np.array(x[n-length:])
        else:
            a = [0]*(length-n)
            x.extend(a)
            return np.array(x)

    def filter_user(self, inter_seq):
        # 此函数用于过滤一些交互过少的用户
        inter_seq['seq_len'] = inter_seq['itemid'].apply(lambda x: len(x))
        user_id = inter_seq[inter_seq['seq_len']>=self.min_item_list_len]['userid']
        return user_id


class DQNDataLoader(torch.utils.data.Dataset):
    # 读取数据
    # 能够从交互数据返回用户编号，用户历史交互序列，有效交互序列长度，路径预测数据划分位置
    def __init__(self, dataset_path):
        super().__init__()
        self.max_item_list_len = 100
        self.min_item_list_len = 3
        # 读入数据
        self.data = pd.read_csv(dataset_path, sep='\t').to_numpy()
        # 利用函数对数据进行处理
        self.interaction = self._load_data(dataset_path)
        self.interaction.rename(columns={'userid':'user_id','itemid':'item_id_list'}, inplace=True)
        self.uid = self.interaction['user_id'].to_numpy()
        self.item_id_list = self.interaction['item_id_list'].to_numpy()
        self.seq_len = self.interaction['seq_len'].to_numpy()
        self.predict_pos = self.interaction['position'].to_numpy()
        self.target = self.interaction['target'].to_numpy()
        self.inter_list = self.interaction['inter_list'].to_numpy()

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, index):
        # 用户编号、用户历史交互序列、有效交互序列长度、路径预测数据划分位置、预测路径最终目标
        return self.uid[index], self.item_id_list[index], self.seq_len[index], self.predict_pos[index], self.target[index], self.inter_list[index]

    def _load_data(self, dataset_path):
        # load data and preprocess it
        columns = ['userid', 'itemid', 'behavior', 'timestamp']
        self.data = pd.read_csv(dataset_path, sep='\t', names=columns)
        # 对每一列数据进行空缺值处理
        for field in columns:
            self.data[field].fillna(value='', inplace=True)
        # 先对数据按照user和时间戳升序排列
        self.data.sort_values(by=['userid','timestamp'], inplace=True, ascending=True)
        self.data.drop_duplicates(subset=['userid','itemid'], keep='first', inplace=True)
        # 再根据用户聚合，得到交互序列和评分序列
        inter_seq = self.data.groupby('userid').agg({'itemid':'unique'}).reset_index()
        # 数据清洗——需要filter一些交互过少的用户,得到保留的
        user_id = self.filter_user(inter_seq)
        # 得到过滤后的交互数据
        new_inter_seq = self.data.groupby('userid').agg({'itemid':'unique'}).reset_index()
        new_inter_seq = new_inter_seq[new_inter_seq['userid'].isin(user_id)]
        new_inter_seq['seq_len'] = new_inter_seq['itemid'].apply(lambda x: len(x))
        new_inter_seq['target'] = new_inter_seq['itemid'].apply(lambda x: x[-1])
        new_inter_seq['inter_list'] = new_inter_seq['itemid'].apply(lambda x: x[:random.randint(max(len(x)-6,0),len(x)-1)])
        # 再得到交互序列长度，以及将数据填充为固定长度list
        new_inter_seq['itemid'] = new_inter_seq['itemid'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))
        # 得到需要预测的路径长度，然后将数据填充为固定长度list
        new_inter_seq['position'] = new_inter_seq['inter_list'].apply(lambda x: len(x))
        new_inter_seq['inter_list'] = new_inter_seq['inter_list'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))

        return new_inter_seq

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
        inter_seq['seq_len'] = inter_seq['itemid'].apply(lambda x: len(x))
        user_id = inter_seq[inter_seq['seq_len']>=self.min_item_list_len]['userid']
        return user_id


class DQNDataLoader_seq(torch.utils.data.Dataset):
    # 读取数据
    # 能够从交互数据返回用户编号，用户历史交互序列，有效交互序列长度，路径预测数据划分位置
    def __init__(self, dataset_path):
        super().__init__()
        self.max_item_list_len = 100
        self.min_item_list_len = 3
        # 读入数据
        self.data = pd.read_csv(dataset_path, sep='\t').to_numpy()
        # 利用函数对数据进行处理
        self.interaction = self._load_data(dataset_path)
        self.interaction.rename(columns={'userid':'user_id','itemid':'item_id_list'}, inplace=True)
        self.uid = self.interaction['user_id'].to_numpy()
        self.item_id_list = self.interaction['item_id_list'].to_numpy()
        self.seq_len = self.interaction['seq_len'].to_numpy()

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, index):
        # 用户编号、用户历史交互序列、有效交互序列长度
        return self.uid[index], self.item_id_list[index], self.seq_len[index]

    def _load_data(self, dataset_path):
        # load data and preprocess it
        columns = ['userid', 'itemid', 'behavior', 'timestamp']
        self.data = pd.read_csv(dataset_path, sep='\t', names=columns)
        # 对每一列数据进行空缺值处理
        for field in columns:
            self.data[field].fillna(value='', inplace=True)
        # 先对数据按照user和时间戳升序排列
        self.data.sort_values(by=['userid','timestamp'], inplace=True, ascending=True)
        self.data.drop_duplicates(subset=['userid','itemid'], keep='first', inplace=True)
        # 再根据用户聚合，得到交互序列和评分序列
        inter_seq = self.data.groupby('userid').agg({'itemid':'unique'}).reset_index()
        # 数据清洗——需要filter一些交互过少的用户,得到保留的
        user_id = self.filter_user(inter_seq)
        # 得到过滤后的交互数据
        new_inter_seq = self.data.groupby('userid').agg({'itemid':'unique'}).reset_index()
        new_inter_seq = new_inter_seq[new_inter_seq['userid'].isin(user_id)]
        new_inter_seq['seq_len'] = new_inter_seq['itemid'].apply(lambda x: len(x))
        # 再得到交互序列长度，以及将数据填充为固定长度list
        new_inter_seq['itemid'] = new_inter_seq['itemid'].apply(lambda x: self.fill_seq(list(x),self.max_item_list_len))

        return new_inter_seq

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
        inter_seq['seq_len'] = inter_seq['itemid'].apply(lambda x: len(x))
        user_id = inter_seq[inter_seq['seq_len']>=self.min_item_list_len]['userid']
        return user_id

class DQNDataLoader_seq2(torch.utils.data.Dataset):
    # 读取数据
    # 能够从交互数据返回用户编号，用户历史交互序列，有效交互序列长度，路径预测数据划分位置
    def __init__(self, dataset_path):
        super().__init__()
        self.max_item_list_len = 100
        self.min_item_list_len = 3
        # 读入数据
        self.interaction = pd.read_csv(dataset_path, sep='\t', header=0,  dtype = {'item_id_list' : str, 'inter_list':str})
        # 利用函数对数据进行处理
        #self.interaction = self.interaction.sample(frac=0.1)
        self.uid = self.interaction['user_id'].to_numpy()
        self.item_id_list = self.interaction['item_id_list'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],self.max_item_list_len)).to_numpy()
        self.seq_len = self.interaction['seq_len'].to_numpy()
        self.predict_pos = self.interaction['position'].to_numpy()
        self.target = self.interaction['target'].to_numpy()
        self.inter_list = self.interaction['inter_list'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],self.max_item_list_len)).to_numpy()

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, index):
        # 用户编号、用户历史交互序列、有效交互序列长度
        return self.uid[index], self.item_id_list[index], self.seq_len[index], self.predict_pos[index], self.target[index], self.inter_list[index]
    
    def fill_seq(self, x, length):
        # 此函数用来将序列补齐为统一长度
        n = len(x)
        if n>=length:
            return x[:length]
        else:
            a = [0]*(length-n)
            x.extend(a)
            return np.array(x)

class DQNDataLoader_seq3(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.interaction = pd.read_csv(dataset_path, sep='\t', header=0)
        #self.interaction = self.interaction.sample(frac=0.1)
        self.uid = self.interaction['user_id'].to_numpy()
        self.inter_len = self.interaction['inter_seq'].apply(lambda x: len([int(i) for i in x.strip('[').strip(']').split(',')])-1).to_numpy()
        self.inter_seq = self.interaction['inter_seq'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],16)).to_numpy()
        self.target = self.interaction['target'].to_numpy()
        self.scene = self.interaction['scene'].to_numpy()
        self.pred_len = self.interaction['pred_path'].apply(lambda x: len([int(i) for i in x.strip('[').strip(']').split(',')])-1).to_numpy()
        self.pred_seq = self.interaction['pred_path'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],16)).to_numpy()

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, index):
        return self.uid[index], self.inter_seq[index], self.inter_len[index], self.scene[index], self.target[index], self.pred_seq[index], self.pred_len[index]
    
    def fill_seq(self, x, length):
        # 此函数用来将序列补齐为统一长度
        n = len(x)
        if n>=length:
            return np.array(x[n-length:])
        else:
            a = [0]*(length-n)
            x.extend(a)
            return np.array(x)