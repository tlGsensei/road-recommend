import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
os.chdir("/workspace/mdls/DFS")
print(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class LSTMPR(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 1, num_layers)
        self.fc1 = nn.Linear(2*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)
        # java 364, net 340, os 500, data_sci 265, computer 310,c_class 116
        self.Embed = nn.Embedding(116, 16)
        self.catEmbed = nn.Embedding(5, 16)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, x_seq, x_cat, target):
        x_seq = self.Embed(x_seq)
        lstm_out,_ = self.lstm(x_seq)
        lstm_out = torch.cat([lstm_out.squeeze(2), self.catEmbed(x_cat)],dim=1)
        fc1_out = self.dropout(self.fc1(lstm_out))
        fc2_out = self.dropout(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)
        logits = torch.matmul(fc3_out,self.Embed.weight.transpose(0,1))
        loss = self.CE(logits, target)
        return loss
    
    def predict(self, x_seq, x_cat):
        x_seq = self.Embed(x_seq)
        lstm_out,_ = self.lstm(x_seq)
        lstm_out = torch.cat([lstm_out.squeeze(2), self.catEmbed(x_cat)],dim=1)
        fc1_out = self.dropout(self.fc1(lstm_out))
        fc2_out = self.dropout(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)
        logits = torch.matmul(fc3_out,self.Embed.weight.transpose(0,1))
        max_val, max_id = logits.max(1)
        return max_id

class trainDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.data = pd.read_csv(dataset_path, sep='\t', header=0)
        self.inter_seq = self.data['inter_seq'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],16)).to_numpy()
        self.cat = self.data['cat'].to_numpy()
        self.target = self.data['target'].to_numpy()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.inter_seq[index], self.cat[index], self.target[index]

    def fill_seq(self, x, length):
        n = len(x)
        if n>=length:
            return np.array(x[n-length:])
        else:
            a = [0]*(length-n)
            x.extend(a)
            return np.array(x)

class testDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.data = pd.read_csv(dataset_path, sep='\t', header=0)
        self.inter_seq = self.data['inter_seq'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],16)).to_numpy()
        self.cat = self.data['cat'].to_numpy()
        self.target = self.data['target'].to_numpy()
        self.pred = self.data['pred_path'].apply(lambda x: self.fill_seq([int(i) for i in x.strip('[').strip(']').split(',')],16)).to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.inter_seq[index], self.cat[index], self.target[index], self.pred[index]

    def fill_seq(self, x, length):
        n = len(x)
        if n>=length:
            return np.array(x[n-length:])
        else:
            a = [0]*(length-n)
            x.extend(a)
            return np.array(x)

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

def train_test(course):
    if course in ['java','net','os','data_sci', 'computer','c_class']:
        path = '/workspace/mdls/RL_path/data/'+course+'/'+course+'_train_10.csv'
    else:
        print('course not supported!')
        return 
    # train_input = trainDataLoader('java_train_same.csv')
    train_input = trainDataLoader(path)
    traindata = torch.utils.data.DataLoader(
        dataset = train_input,
        batch_size = 64,
        shuffle = True
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMPR(input_size=16,hidden_size=16,num_layers=2,output_size=16).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    for epoch in range(50):
        train_loss = 0
        for i, data in enumerate(traindata):
            in_seq = data[0].to(device)
            in_cat = data[1].to(device)
            target = data[2].to(device)
            optimizer.zero_grad()
            loss = model(in_seq, in_cat, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if epoch%10 == 0:
            print("epoch ",epoch, " train_loss: ",train_loss/i)
    print("train finished!")

    model.eval()
    test_path = '/workspace/mdls/RL_path/data/'+course+'/'+course+'_test_10.csv'
    # test_input = testDataLoader('java_test_same.csv')
    test_input = testDataLoader(test_path)
    testdata = torch.utils.data.DataLoader(
        dataset = test_input,
        batch_size = 1,
        shuffle = True
    )
    p_list = []
    r_list = []
    f_list = []
    for i, data in enumerate(testdata):
        in_seq = data[0].to(device)
        in_cat = data[1].to(device)
        target = data[2].to(device)
        true_path = data[3].to(device)[0].to('cpu')
        pred_path = []
        step = 0
        while True:
            output = model.predict(in_seq, in_cat)
            pred_path.append(output[0].to('cpu'))
            step += 1
            if step>10 or target == output:
                break
        p,r,f = cal_p_r(pred_path, true_path)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    #print("precision_list ",p_list)
    #print("recall_list ",r_list)
    #print("f1-score ",f_list)
    print("precision: ", np.array(p_list).mean(), "recall: ", np.array(r_list).mean(), "f1-score: ", np.array(f_list).mean())
    print('完成测试！')
    return np.array(p_list).mean(), np.array(r_list).mean(), np.array(f_list).mean()

all_p = []
all_r = []
all_f = []
for i in range(5):
    p,r,f = train_test('c_class')
    all_p.append(p)
    all_f.append(f)
    all_r.append(r)
print("平均p: ", np.array(all_p).mean(),"平均r: ", np.array(all_r).mean(),"平均f1: ", np.array(all_f).mean())
file = open('LSTMPR_results.txt','a')
file.write("c_class 5次不同训练，每次训练取50个epoch，取平均\n")
file.write("precision: "+str(all_p)+'\n'+"recall: "+str(all_r)+'\n'+"f1 score: "+str(all_f)+'\n')
file.write('平均precison： '+str(np.array(all_p).mean())+" 平均recall: "+str(np.array(all_r).mean())+" 平均f1: "+str(np.array(all_f).mean())+'\n')
file.close()