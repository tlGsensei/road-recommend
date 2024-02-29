1 train_k.py 设置不同路径长度，完成学习路径训练+预测
切换课程时，更改函数user_model_train()中课程代码,注意网院和头歌embsize不一样在注释有写，需要随课程更改
           更改for循环中，结果文件写入路径
           for循环中，i用于控制路径预测长度，第二个for循环用于控制epoch
           强化学习结果特别不稳定，因此，取多个epoch的平均
当前预测结果，如java_result_to10_5.txt 记录了1-10的路径长度，每个5次迭代的平均

2 train_ablation.py 消融实验，控制不同模块是否使用，以及emb大小
    奖励函数消融实验时，更改model.py文件里面reward计算方式，unit_reward表示知识点奖励函数，path_reward表示序列级奖励函数，通过注释的方式控制变量
    用户模型消融实验时，user_model_train 里面更改注释和非注释部分，path_env表示带user_model的，path_env_without_user_model表示不带用户模型的
    控制emb_size时，注意与训练的知识图谱向量是否保持一致

3 data文件夹下存放了处理好的数据，处理代码，以及知识图谱预训练的Embedding

4 saved文件夹下存放了模型文件

5 system_test.py 系统测试数据

6 system_result.ipynb 系统测试数据结果画图，png箱线图就由此得来