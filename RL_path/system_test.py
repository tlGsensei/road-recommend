import sys
from turtle import position
from tqdm import tqdm
sys.path.append('../../')

import pandas as pd
import numpy as np
import torch
from torch import nn
from dataloader import *
from model import *
from utils import *
import os
os.chdir("/workspace/mdls/RL_path")
print(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def user_model_train(course, device, BATCH_SIZE,emb_dim):
    if course == 'java':
        path = './data/java/java_mb_KG.txt'
        item_emb_path = './data/java/transe_'+str(emb_dim)+'.ckpt'
        # item_emb_path = './data/java/transe.ckpt'
        size = 364
    elif course == 'net':
        path = './data/net/net_mb_KG.txt'
        item_emb_path = './data/net/transe.ckpt'
        size = 338
    elif course == 'os':
        path = './data/os/os_mb_KG.txt'
        item_emb_path = './data/os/transe.ckpt'
        size = 496
    elif course == 'c_class':
        path = './data/c_class/c_class_mb.txt'
        item_emb_path = './data/c_class/transe.ckpt'
        size = 114
    elif course == 'computer':
        path = './data/computer/computer_mb.txt'
        item_emb_path = './data/computer/transe.ckpt'
        size = 307
    elif course == 'data_sci':
        path = './data/data_sci/data_sci_mb.txt'
        item_emb_path = './data/data_sci/transe.ckpt'
        size = 263
    else:
        print('course not supported!')
        return
    train_input = SeqDataLoader(path)

    traindata = torch.utils.data.DataLoader(
        dataset = train_input,
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    # wangyuan 310
    # touge 8750
    # model = path_env_without_user_model(input_size=emb_dim, hidden_size=emb_dim, num_layers=3, n_users=310, emb_size=emb_dim, item_emb_path=item_emb_path, device=device)
    model = path_env(input_size=emb_dim, hidden_size=emb_dim, num_layers=3, n_users=8750, emb_size=emb_dim, item_emb_path=item_emb_path, device=device)
    # n_items 代表序列最长长度
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    
    trainloss_res = []
    for epoch in range(500):

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
            loss = model(uid, item_id_list, seq_len, target)
            #loss = model.loss_func(output, target)
            all_train_loss += loss
            loss.backward()
            optimizer.step()
        if (epoch+1)%10 == 0:
            # print("training:",epoch,'trainloss:',all_train_loss/(batch_idx+1))
            trainloss_res.append(all_train_loss.item()/(batch_idx+1))
    #print(trainloss_res)
    return model

def dqn_train(cfg, env, course, k, emb_dim):
    print('DQN开始训练!')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')

    if course == 'java':
        path = './data/java/java_train_10.csv'
        size = 363
    elif course == 'net':
        path = './data/net/net_train_10.csv'
        size = 338
    elif course == 'os':
        path = './data/os/os_train_10.csv'
        size = 496
    elif course == 'c_class':
        path = './data/c_class/c_class_train_10.csv'
        size = 114
    elif course == 'computer':
        path = './data/computer/computer_train_10.csv'
        size = 307
    elif course == 'data_sci':
        path = './data/data_sci/data_sci_train_10.csv'
        size = 263
    else:
        print('course not supported!')
        return

    train_input = DQNDataLoader_seq3(path)

    traindata = torch.utils.data.DataLoader(
        dataset = train_input,
        batch_size = cfg.batch_size,
        shuffle = True
    )

    env.to(cfg.device)
    agent = DQN(state_dim=emb_dim*3, action_dim=size,cfg=cfg)

    rewards = [] # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    # all_loss = []
    for i_ep in tqdm(range(cfg.train_eps)):
        # 改为dataloader加载数据的
        for batch_idx, data in enumerate(traindata):
            # 每个user逐个生成序列去计算
            #print(batch_idx)
            uid = data[0].to(cfg.device)
            inter_seq = data[1].to(cfg.device)
            inter_len = data[2].to(cfg.device)
            scene = data[3].to(cfg.device)
            target = data[4].to(cfg.device)
            pred_seq = data[5].to(cfg.device)
            pred_len = data[6].to(cfg.device)
            env.set_for_user(
                uid = uid,
                inter_seq = inter_seq, 
                inter_len = inter_len, 
                scene = scene, 
                target = target, 
                pred_seq = pred_seq,
                pred_len = pred_len
            )
            state = env.set_state()
            #done = False
            ep_reward = 0
            step = 0
            # all_loss = torch.tensor(0).to(cfg.device)
            # torch.autograd.set_detect_anomaly(True)
            while True:
                step += 1
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                agent.memory.push(state, action, reward, next_state, done)
                #agent.update()
                # loss = agent.update_batch(state,action,next_state,reward,done)
                # agent.update_batch(state,action,next_state,reward,done)
                # all_loss.append(loss.item())
                # all_loss = all_loss + loss
                state = next_state
                if done.any() or step==k:
                    break
            agent.optimizer.zero_grad()
            # all_loss.backward()
            loss = agent.update_batch(state,action,next_state,reward,done)
            loss.backward()
            agent.optimizer.step()
            if (i_ep+1) % cfg.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            # save ma_rewards
            '''
            if ma_rewards:
                ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
            else:
                ma_rewards.append(ep_reward)
            '''
            #print("reward:", reward)
    print('dqn完成训练！')
    #agent.save('./saved/java_train_'+str(k))
    #print(all_loss)
    return rewards, ma_rewards,agent

def dqn_eval(cfg,env,agent,course,k):
    print('DQN开始测试!')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')

    if course == 'java':
        path = './data/java/java_test_10.csv'
    elif course == 'net':
        path = './data/net/net_test_10.csv'
    elif course == 'os':
        path = './data/os/os_test_10.csv'
    elif course == 'c_class':
        path = './data/c_class/c_class_system.csv'
        size = 114
    elif course == 'computer':
        path = './data/computer/computer_system.csv'
        size = 307
    elif course == 'data_sci':
        path = './data/data_sci/data_sci_system.csv'
        size = 263
    else:
        print('course not supported!')
        return

    test_input = DQNDataLoader_seq3(path)

    testdata = torch.utils.data.DataLoader(
        dataset = test_input,
        batch_size = cfg.batch_size,
        shuffle = True
    )

    p_list = []
    r_list = []
    f_list = []
    for batch_idx, data in enumerate(testdata):
        uid = data[0].to(cfg.device)
        inter_seq = data[1].to(cfg.device)
        inter_len = data[2].to(cfg.device)
        scene = data[3].to(cfg.device)
        target = data[4].to(cfg.device)
        pred_seq = data[5].to(cfg.device)
        pred_len = data[6].to(cfg.device)
        env.set_for_user(
            uid = uid,
            inter_seq = inter_seq, 
            inter_len = inter_len, 
            scene = scene, 
            target = target, 
            pred_seq = pred_seq,
            pred_len = pred_len
        )
        state = env.set_state()
        #done = False
        ep_reward = 0
        step = 0
        true_path = pred_seq[0].to('cpu')
        pred_path = []
        while True:
            step += 1
            action = agent.predict(state,pred_path)
            # action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            pred_path.append(action.item())
            if done.any() or step==k:
                break
        #print("predicted: ",pred_path)
        #print("true: ",true_path.cpu())
        p,r,f = cal_p_r(pred_path, true_path)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    #print("precision_list ",p_list)
    #print("recall_list ",r_list)
    #print("f1-score ",f_list)
    print("k setting: ",k)
    print("precision: ", np.array(p_list).mean(), "recall: ", np.array(r_list).mean(), "f1-score: ", np.array(f_list).mean())
    return p_list,r_list,f_list,np.array(p_list).mean(),np.array(r_list).mean(),np.array(f_list).mean()

c_name = 'c_class'
emb_dim = 16

import setproctitle
setproctitle.setproctitle("mdls_dqn_"+c_name)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# dev = 'cpu'
env = user_model_train(course=c_name, device=dev, BATCH_SIZE=32, emb_dim=emb_dim)
configs = DQNConfig()
configs.train_eps=1
configs.batch_size=1
# configs.device = 'cpu'
#env.device = 'cpu'
#dqn_train(cfg=configs, env= env.cpu(), course='java')
for i in range(1,6):
    rewards,_,agent = dqn_train(cfg=configs, env= env, course=c_name,k=i, emb_dim=emb_dim)
    _,_,_,p,r,f = dqn_eval(configs,env,agent,course=c_name,k=i)
    # file = open(c_name+'_result_to10_5.txt','a')
    # file.write("路径长度 "+str(i)+' precison： '+str(p)+" recall: "+str(r)+" f1: "+str(f)+'\n')
    # file.close()
    print('dim = ',emb_dim)
    print("with user model")
    print(c_name, ": precsion: ", p," recall: ",r," f1 score: ",f)


