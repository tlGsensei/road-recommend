from path_model import *

def class3_path(user_id, seq, target):
    saved_3_path = './savedir/class3_seq.pth'
    return load(saved_3_path, user_id, seq, target)

def class4_path(user_id, seq, target):
    saved_4_path = './savedir/class4_seq.pth'
    return load(saved_4_path, user_id, seq, target)

'''
saved_path = './savedir/class3_seq.pth'
user_id = 17762
seq = [1,4,3,5]
target = 9
path = load(saved_path, user_id, seq, target)
print(path)
'''

def class4_path2(user_id, scene, seq, target):
    saved_4_path = './savedir/class4_seq.pth'
    class4_shixun_dict = {'zlg2nmcf':1,'3ozvy5f8':2,'b6ljcet3':3,'mbgfitn6':4,'uc64f2qs':5,'vtnag4op':6,'w3vcokrg':7,'uywljq4v':8,'ba56rk8v':9}
    reverse_dict = {v: k for k, v in class4_shixun_dict.items()}
    seq_id = [class4_shixun_dict[x] for x in seq]
    target_id = class4_shixun_dict[target]
    path_id = load_with_scene(saved_4_path, scene, user_id, seq_id, target_id)
    path = [reverse_dict[x] for x in path_id]
    return path

print('user1:初始：',class4_path2(1,'初始',['zlg2nmcf','mbgfitn6','b6ljcet3'],'uc64f2qs'))
print('user1:复习：', class4_path2(1,'复习',['zlg2nmcf','mbgfitn6','b6ljcet3'],'uc64f2qs'))
print('user1:考前: ',class4_path2(1,'考前',['zlg2nmcf','mbgfitn6','b6ljcet3'],'uc64f2qs'))
print('user100:初始: ',class4_path2(100,'初始',['zlg2nmcf','mbgfitn6','b6ljcet3'],'uc64f2qs'))
print('user100:复习: ', class4_path2(100,'复习',['zlg2nmcf','mbgfitn6','b6ljcet3'],'uc64f2qs'))
print('user100:考前: ', class4_path2(100,'考前',['zlg2nmcf','mbgfitn6','b6ljcet3'],'uc64f2qs'))